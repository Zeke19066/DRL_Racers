import multiprocessing
import random
import os
import sys
import time
import cv2
import numpy as np

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Game_Interface
from pynput.keyboard import Key, Listener # for training reward bonuses.

'''
Notes: Current build spawns pointer on the edge of an invisible box determined by 'box_radius'.
The agent can then move the pointer within the range of box_radius, with moves outside being negated.
Once box_radius reaches 80px, spawning is unchanged but agen can move freely.
Rewards that take more than box_radius/2 moves to reach will be progressively discounted.
'''

# Neural Network Parameters & Hyperparameters
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 5 # How many output acitons?
        self.flat_size = 222784 #size of the flattened input image (out.shape to get size)

        self.conv1 = nn.Conv2d(4, 32, 8, 4) #in_channels, out_channels, kernel_size, stride, padding
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(self.flat_size, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)
        self.softmax = nn.Softmax(dim=-1)

    # Forwardpass Method
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        #print(f'Flattened Shape: {out.shape}') #This needs to be calculater as this image shape needs to feed into next layer.
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.softmax(out)
        return out

class ModelTrain():

    def __init__(self):
        print('Initializing Training')
        self.parameters = {
            'crash_counter': 4,
            'gamma': 0.99,
            'final_epsilon': 0,
            'initial_epsilon': 0,
            'number_of_iterations': 150000,
            'replay_memory_size': 400, #500 is the hard limit running AoM on RTX 2070 Super
            'minibatch_size': 50,
            'save_count': 500, # How many cycles before we save?
            'key_bonuses': [0, 0, 0, 0, 0, 0, 0], # User selected reward bonuses will be stored here.
            'win_count': 0, 'fail_count': 0, 'performance_trend': 0,
            'iteration': 0,
            'back_on_track_bonus': 0
            }

    # Training Parameters
    def train(self, model, start, send_connection):
        print('Initializing Training Method')

        '''
        # Multithread Reward Bonus Listener; won't impact performance
        listener = Listener(on_press=self.on_press)
        listener.start()
        '''

        #store tracking variables.
        total_reward, self.reset_counter, self.last_reset = 0, 0, 0 # The action counters
        performance, performance_last = [], [1] #can't have a zero denominator
        screen = Game_Interface.ScreenGrab()
        controller = Game_Interface.Actions()
        replay_memory = [] # initialize replay memory

        optimizer = optim.Adam(model.parameters(), lr=1e-6) # define Adam optimizer; lr=3e-4 (0.0003) learning rate too fast & unstable
        criterion = nn.MSELoss() # initialize mean squared error loss
        action = torch.zeros([model.number_of_actions], dtype=torch.float32) # Generate a matrix of zeroes
        action[0] = 1 #the first (intialization) action is to do nothing.
        
        controller.Reset() # Reset the game state for a clean start.

        #Call to the resize & image to tensor methods.
        image_data = screen.quick_Grab()
        image_data = screen.add_Mouse(image_data) #refacor this!
        image_data = self.image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)
        # You can use torch.cat to concatenate a sequence of tensors along a given dimension

        # initialize epsilon value
        epsilon = self.parameters['initial_epsilon']
        epsilon_decrements = np.linspace(self.parameters['initial_epsilon'], self.parameters['final_epsilon'], self.parameters['number_of_iterations'])

        # main infinite loop
        print('Entering Training Loop')
        while self.parameters['iteration'] < self.parameters['number_of_iterations']:            
            
            move_reward, random_check = 0, False
            self.reset_counter = self.parameters['iteration'] - self.last_reset

            # get output & initialize action
            output = model(state)[0]
            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action = action.cuda()

            # epsilon greedy exploration
            random_action = random.random() <= epsilon
            if random_action:
                random_check = True
                #print("_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_* RANDOM _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*")
            action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                            if random_action
                            else torch.argmax(output)][0]
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action_index = action_index.cuda()
            action[action_index] = 1 # This sets max to 1

            # Get Action, Execute, and get ScreenState
            np_actions = action.cpu().detach().numpy() # Converting from CUDA tensor to NP Array
            np_action = np.where(np_actions == np.amax(np_actions)) # Taking the index with the highest activation
            np_action = np_action[0]
            controller.action_Map(np_action) # Execute Action
            image_data_1 = screen.quick_Grab() # Get the next screenshot

            # Local reward rules
            minmap_reward = screen.minmap_Scan(image_data_1) # Get Reward

            if int(np_action[0]) == 2:
                minmap_reward = -1

            if minmap_reward > 0:
                if self.parameters['back_on_track_bonus'] >= 5: #Back on track boost.
                    minmap_reward += 3
                    print('BACK ON TRACK +3 BOOST')
                    self.parameters['back_on_track_bonus'] = 0
                self.parameters['back_on_track_bonus'] = 0
                self.parameters['win_count'] += 1

            elif minmap_reward < 0:
                self.parameters['back_on_track_bonus'] += 1
                self.parameters['fail_count'] += 1
        
            side_set = (3,4)
            if self.reset_counter < 4:  # prevent the corner glitch exploitation.
                if int(np_action[0]) in side_set:
                    minmap_reward = -2
                    self.parameters['fail_count'] = 0
                    self.last_reset = self.parameters['iteration']
                    controller.Reset() # Reset the game state for a clean start.

            if self.parameters['fail_count'] >= 50: # Trigger reset
                performance.append(total_reward) #let's track how we did.
                total_reward = 0
                self.parameters['back_on_track_bonus'] = 0
                self.parameters['fail_count'] = 0
                minmap_reward = -2 #since agent failed, we want negative association with state.
                self.last_reset = self.parameters['iteration']
                controller.Reset() # Reset the game state for a clean start.

            move_reward += self.parameters['key_bonuses'][int(np_action)] # Add in any keybonuses
            reward = minmap_reward + move_reward
            total_reward += reward
            total_reward = round(total_reward, 4)

            # Prepare and Submit Screenshot through forward pass.
            image_data_1 = screen.add_Mouse(image_data_1) #REFACTOR ME!
            image_data_1 = self.image_to_tensor(image_data_1)
            state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            # Execute Batching & Optimization
            replay_memory.append((state, action, reward, state_1)) # save transition to replay memory
            if len(replay_memory) > self.parameters['replay_memory_size']:# if replay memory is full, remove the oldest transition
                replay_memory.pop(0)
            epsilon = epsilon_decrements[self.parameters['iteration']] # epsilon annealing
            minibatch = random.sample(replay_memory, min(len(replay_memory), self.parameters['minibatch_size'])) # sample random minibatch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()
            output_1_batch = model(state_1_batch) # get output for the next state
            #set y_j to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] + self.parameters['gamma'] * torch.max(output_1_batch[i])
                                    for i in range(len(minibatch))))
            q_value = torch.sum(model(state_batch) * action_batch, dim=1) # extract Q-value
            optimizer.zero_grad()  # PyTorch accumulates gradients by default, so they need to be reset in each pass
            y_batch = y_batch.detach() # returns a new Tensor, detached from the current graph, the result will never require gradient
            loss = criterion(q_value, y_batch) # calculate loss
            loss.backward() 
            optimizer.step()
            state = state_1
            self.parameters['iteration'] += 1


            #Multiprocessing pipe to send metrics to GUI
            qmax = round(np.max(output.cpu().detach().numpy()),2)
            try:
                metrics = [int(np_action[0]), repeat_count, total_reward, (self.parameters['iteration']-self.reset_count),
                        (round(epsilon, 3)), self.parameters['iteration'], round((time.time() - start), 1), qmax,
                        random_check, self.parameters['performance_trend'], controller.box_size, self.parameters['win_count'], self.parameters['fail_count']]
                self.gui_send(send_connection, metrics)
            except:
                pass

            '''
            if self.parameters['iteration'] % 500 == 0: #Check in on the batch performance.
                if np.mean(performance) < np.mean(performance_last):
                    performance = []
                    replay_memory = [] # empty replay memory
                    print('********PURGING REPLAY MEMORY********')

                elif np.mean(performance) >= np.mean(performance_last):
                    performance_last = performance
                    performance = []
                    print(f'********KEEPING MEMORY:{performance_last}********')
            '''

            if self.parameters['iteration'] % self.parameters['save_count'] == 0: # Save model parameters if it is performing. Otherwise restart
                torch.save(model.state_dict(), "C:/Users/Ezeab/Documents/Python/DRL_Racers/pretrained_model/current_model_output.pth")
            cycle = self.parameters['iteration']
            print(f'Reward:{minmap_reward} cycle:{cycle} time:{round((time.time() - start),1)} e:{round(epsilon,3)}, total reward:{total_reward}, Q max:{qmax}')

    # Testing Environment
    def test(self, model):

        #game_state = GameState()

        # initial action is do nothing
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action[0] = 1

        image_data, reward, terminal = game_state.frame_step(action)
        image_data = resize_and_rgb2gray(image_data)
        image_data = image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        while True:
            # get output from the neural network
            output = model(state)[0]

            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action = action.cuda()

            # get action
            action_index = torch.argmax(output)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action_index = action_index.cuda()
            action[action_index] = 1

            # get next state
            image_data_1, reward, terminal = game_state.frame_step(action)
            image_data_1 = resize_and_rgb2gray(image_data_1)
            image_data_1 = image_to_tensor(image_data_1)
            state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

            # set state to be state_1
            state = state_1


    # A method for communicating with the GUI through Multiprocess Pipe
    def gui_send(self, conn, metrics):
        metrics.append(-9999) # just some fodder to signal the end of the transmission.
        for metric in metrics:
            conn.send(metric)
        #conn.close()


    # A method for assigning a reward bonus on the fly with the WASD keys.
    def on_press(self, key):
        key_to_num = ('z', 'e', 'q', 'a', 'd', 'w', 's')
        str_key = key.char
        for i, n in enumerate(key_to_num):
            if n == str_key:
                self.parameters['key_bonuses'][i] += 0.1
                print_keys = self.parameters['key_bonuses'] #we need to assign the variable in order to print.
                print(f'{key} pressed   i:{i}  {print_keys}')
        time.sleep(0.2)


    def image_to_tensor(self, image):
        image_tensor = image.transpose(2, 0, 1)
        #print(image_tensor.shape)
        image_tensor = image_tensor.astype(np.float32)
        image_tensor = torch.from_numpy(image_tensor) # Creates a Tensor from a numpy.ndarray (Nth Dimension Array).
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            image_tensor = image_tensor.cuda()
        return image_tensor

# Intialization weights
def initial_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

# Script initialization switchboard
def main(mode, send_connection, first_run=False):
    if mode == 'test':
        cuda_is_available = torch.cuda.is_available()
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()
        test(model)

    elif mode == 'train':
        cuda_is_available = torch.cuda.is_available()
        '''
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
        '''
        while 1 > 0:
            model = NeuralNetwork()
            if not first_run:
                model.load_state_dict(torch.load('C:/Users/Ezeab/Documents/Python/DRL_Racers/pretrained_model/current_model_seed.pth'))
                #model.eval() use this in the evalution (testing) not training.
            if cuda_is_available:  # put on GPU if CUDA is available
                model = model.cuda()
            if first_run:
                model.apply(initial_weights) #only 
            start = time.time()

            choo_choo = ModelTrain()
            choo_choo.train(model, start, send_connection)

if __name__ == "__main__":
    main('train', False) #Local initialization won't use the GUI
