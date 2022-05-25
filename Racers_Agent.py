"""
How it works:
Everyone uses the same conv layers for feature extraction, and the split between agent and critic only happens in the last layer.
The initialization weights look a little wonky as well, but it might just be more modern.

This is called from train()



"We can combine the actor and critic losses if we want using a discount factor to bring them 
to the same order of magnitude. Adding an entropy term is optional, but it encourages our 
actor model to explore different policies and the degree to which we want to experiment can be 
controlled by an entropy beta parameter."

"""
from os.path import exists
import os
import numpy as np
import torch as T
import torch.optim as optim
from torch.distributions.categorical import Categorical

from collections import deque

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#The PPO model structure
class PPONetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, alpha, file_name='pretrained_model/current_model_seed.pth'):
        super(PPONetwork, self).__init__()
        
        self.checkpoint_file = file_name
        self.number_of_actions = 5 # How many output acitons?
        self.flat_size = 9216

        self.conv1 = nn.Conv2d(4, 32, 8, 4) #in_channels, out_channels, kernel_size, stride, padding
        #self.conv1.requires_grad = False
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        #self.conv2.requires_grad = False
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        #self.conv3.requires_grad = False

        self.linear = nn.Linear(self.flat_size, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self.home_dir = os.getcwd()

        file_exists = exists(self.checkpoint_file)
        if file_exists:
            self.load_checkpoint(self.checkpoint_file)
        else:
            self._initialize_weights()
            self.save_checkpoint(self.checkpoint_file)

        self.optimizer = optim.Adam(self.parameters(), alpha) # define Adam optimizer;
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.linear(out.view(out.size(0), -1))
        #print(f'Flattened Shape: {out.shape}') #This needs to be calculater as this image shape needs to feed into next layer.
        return self.actor_linear(out), self.critic_linear(out)

    def save_checkpoint(self, file_name='pretrained_model/current_model_seed.pth'):
        os.chdir(self.home_dir) #make sure we're in the main folder
        T.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name='pretrained_model/current_model_seed.pth'):
        os.chdir(self.home_dir) #make sure we're in the main folder
        self.load_state_dict(T.load(file_name)) #strict=False

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        #only one element tensors can be converted to Python scalars; np.array(self.states)
        return  np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class NNMerge:

    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict


    def merge_models(self):
        self.parent_1 = PPONetwork(4, 5, self.parameters_dict["learning_rate"], file_name='pretrained_model/Merge_Folder/parent1.pth')
        self.parent_2 = PPONetwork(4, 5, self.parameters_dict["learning_rate"], file_name='pretrained_model/Merge_Folder/parent2.pth')
        self.parent_3 = PPONetwork(4, 5, self.parameters_dict["learning_rate"], file_name='pretrained_model/Merge_Folder/parent3.pth')
        self.parent_4 = PPONetwork(4, 5, self.parameters_dict["learning_rate"], file_name='pretrained_model/Merge_Folder/parent4.pth')
        self.parent_5 = PPONetwork(4, 5, self.parameters_dict["learning_rate"], file_name='pretrained_model/Merge_Folder/parent5.pth')
        self.merge_output = PPONetwork(4, 5, self.parameters_dict["learning_rate"], file_name='pretrained_model/current_model_seed.pth')
    
        #convert tensors to numpy arrays
        np_parent_1 = self.model2numpy(self.parent_1)
        np_parent_2 = self.model2numpy(self.parent_2)
        np_parent_3 = self.model2numpy(self.parent_3)
        np_parent_4 = self.model2numpy(self.parent_4)
        np_parent_5 = self.model2numpy(self.parent_5)
        np_merge_output = []

        for n in range(len(np_parent_1)): #for every item
            average_array = (np_parent_1[n]+np_parent_2[n]+np_parent_3[n]+np_parent_4[n]+np_parent_5[n])/5
            np_merge_output.append(average_array) #append the average of the parents.

        #now convert nump array output to tensors in model.
        self.numpy2model(np_merge_output)

    def model2numpy(self, model):
        #converts the model tensors to NP arrays for easier merging.
        np_model = [] #initialize the holder

        ##First do the weights 
        np_conv1_weights = model.conv1.weight.detach().cpu().numpy().astype('float32')
        np_model.append(np_conv1_weights)
        np_conv2_weights = model.conv2.weight.detach().cpu().numpy().astype('float32')
        np_model.append(np_conv2_weights)
        np_conv3_weights = model.conv3.weight.detach().cpu().numpy().astype('float32')
        np_model.append(np_conv3_weights)
        np_linear_weights = model.linear.weight.detach().cpu().numpy().astype('float32')
        np_model.append(np_linear_weights)
        np_critic_linear_weights = model.critic_linear.weight.detach().cpu().numpy().astype('float32') 
        np_model.append(np_critic_linear_weights)
        np_actor_linear_weights = model.actor_linear.weight.detach().cpu().numpy().astype('float32')
        np_model.append(np_actor_linear_weights)

        ##Now do biases
        np_conv1_bias = model.conv1.bias.detach().cpu().numpy().astype('float32')
        np_model.append(np_conv1_bias)
        np_conv2_bias = model.conv2.bias.detach().cpu().numpy().astype('float32')
        np_model.append(np_conv2_bias)
        np_conv3_bias = model.conv3.bias.detach().cpu().numpy().astype('float32')
        np_model.append(np_conv3_bias)
        np_linear_bias = model.linear.bias.detach().cpu().numpy().astype('float32')
        np_model.append(np_linear_bias)
        np_critic_linear_bias = model.critic_linear.bias.detach().cpu().numpy().astype('float32') 
        np_model.append(np_critic_linear_bias)
        np_actor_linear_bias = model.actor_linear.bias.detach().cpu().numpy().astype('float32')
        np_model.append(np_actor_linear_bias)

        return np_model

    def numpy2model(self, np_model):

        ##First do the weights 
        tensor_conv1_weights = T.tensor(np_model[0], dtype=T.float)
        self.merge_output.conv1.weight = nn.Parameter(data=tensor_conv1_weights, requires_grad=True)
        tensor_conv2_weights = T.tensor(np_model[1], dtype=T.float)
        self.merge_output.conv2.weight = nn.Parameter(data=tensor_conv2_weights, requires_grad=True)
        tensor_conv3_weights = T.tensor(np_model[2], dtype=T.float)
        self.merge_output.conv3.weight = nn.Parameter(data=tensor_conv3_weights, requires_grad=True)
        tensor_linear_weights =  T.tensor(np_model[3], dtype=T.float)
        self.merge_output.linear.weight = nn.Parameter(data=tensor_linear_weights, requires_grad=True)
        tensor_critic_linear_weights = T.tensor(np_model[4], dtype=T.float)
        self.merge_output.critic_linear.weight = nn.Parameter(data=tensor_critic_linear_weights, requires_grad=True)
        tensor_actor_linear_weights = T.tensor(np_model[5], dtype=T.float)
        self.merge_output.actor_linear.weight = nn.Parameter(data=tensor_actor_linear_weights, requires_grad=True)
        
        tensor_conv1_bias = T.tensor(np_model[6], dtype=T.float)
        self.merge_output.conv1.bias = nn.Parameter(data=tensor_conv1_bias, requires_grad=True)
        tensor_conv2_bias = T.tensor(np_model[7], dtype=T.float)
        self.merge_output.conv2.bias = nn.Parameter(data=tensor_conv2_bias, requires_grad=True)
        tensor_conv3_bias = T.tensor(np_model[8], dtype=T.float)
        self.merge_output.conv3.bias = nn.Parameter(data=tensor_conv3_bias, requires_grad=True)
        tensor_linear_bias =  T.tensor(np_model[9], dtype=T.float)
        self.merge_output.linear.bias = nn.Parameter(data=tensor_linear_bias, requires_grad=True)
        tensor_critic_linear_bias = T.tensor(np_model[10], dtype=T.float)
        self.merge_output.critic_linear.bias = nn.Parameter(data=tensor_critic_linear_bias, requires_grad=True)
        tensor_actor_linear_bias = T.tensor(np_model[11], dtype=T.float)
        self.merge_output.actor_linear.bias = nn.Parameter(data=tensor_actor_linear_bias, requires_grad=True)
        
        print('... Merge complete: saving model ...')
        self.merge_output.save_checkpoint(file_name='pretrained_model/current_model_seed.pth')

class Agent:
    def __init__(self, parameters_dict):

        # gamma=0.99, alpha=0.0003, gae_lambda=0.95,
        #   policy_clip=0.2, batch_size=64, n_epochs=10

        self.gamma = parameters_dict["gamma"]
        self.policy_clip = parameters_dict["epsilon"]
        self.n_epochs = parameters_dict["epochs"]
        self.gae_lambda = parameters_dict["tau"]
        self.beta = parameters_dict["beta"]
        self.parameters_dict = parameters_dict
        self.critic_discount = 0.0225 #0.05, 0.1 #scale the critic values
        self.critic_discount_targets = []

        self.initial_state_bool = True
        self.state_tensor = [] #initialize empty.
        self.state_array = []
        self.complex_buffer = deque(maxlen=30) #we average 30 cycles/second;yields 1 second reference window to sample.

        self.model = PPONetwork(4, 5, parameters_dict["learning_rate"])
        self.memory = PPOMemory(parameters_dict["batch_size"])
        self.merge = NNMerge(parameters_dict) #For merging NN
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, file_name='pretrained_model/current_model_seed.pth'):
        print('... saving model ...')
        self.model.save_checkpoint(file_name)

    def load_models(self, file_name='pretrained_model/current_model_seed.pth'):
        print('... loading model ...')
        self.model.load_checkpoint(file_name)

    def merge_models(self):
        self.merge.merge_models()

    def choose_action(self, state):
        #state = T.tensor([observation], dtype=T.float).to(self.model.device)
        state = T.tensor(state, dtype=T.float).to(self.model.device)
        state = state.squeeze(0)
        state = state.unsqueeze(0)
        #if T.cuda.is_available():  # put on GPU if CUDA is available
        #    state = state.cuda()

        action_space, value = self.model(state)
        actions_distribution = F.softmax(action_space, dim=1)
        actions_distribution = Categorical(actions_distribution)

        action = actions_distribution.sample()

        probs = T.squeeze(actions_distribution.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value, action_space

    def learn(self):

        #subfunction for so len(data)%batch size = 0; evenly trims for zero remainder.
        #even though the mask is generated each time, its set to len&divisor. Same len&divisor = same mask every time.
        def array_adjust(array, divisor):
            input_len = len(array)
            remainder = input_len%divisor
            #if there is no remainder, we return the original array.
            if remainder != 0:#create a mask of what indexies to remove.
                mask_rate = np.floor(input_len/remainder)
                advantage = np.zeros(remainder, dtype=np.int32) #initialize advantage array as all zeros.
                for i, val in enumerate(advantage):
                    advantage[i] = i*mask_rate
                #now apply the mask
                for val in advantage:
                    array.pop(val)
            return array
        
        self.memory.batch_size = int(len(self.memory.states) /5) #take the floor
        #all arrays have same length
        self.memory.states = array_adjust(self.memory.states, self.memory.batch_size)
        self.memory.actions = array_adjust(self.memory.actions, self.memory.batch_size)
        self.memory.probs = array_adjust(self.memory.probs, self.memory.batch_size)
        self.memory.vals = array_adjust(self.memory.vals, self.memory.batch_size)
        self.memory.rewards = array_adjust(self.memory.rewards, self.memory.batch_size)
        self.memory.dones = array_adjust(self.memory.dones, self.memory.batch_size)

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr #Array of critic values
            advantage = np.zeros(len(reward_arr), dtype=np.float32) #initialize advantage array as all zeros.

            for t in range(len(reward_arr)-1): #t = timestep; for every timestep in the reward array.
                discount = 1
                a_t = 0 # advantage at timestep initialized as zero.
                for k in range(t, len(reward_arr)-1): #k= timesteps to the end; for every timestep in the future.
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.model.device)
            values = T.tensor(values).to(self.model.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.model.device)

                old_probs = T.tensor(old_prob_arr[batch]).to(self.model.device)
                actions = T.tensor(action_arr[batch]).to(self.model.device)

                actions_distribution, critic_value = self.model(states)
                actions_distribution = F.softmax(actions_distribution, dim=1)
                actions_distribution = Categorical(actions_distribution)
                new_m = actions_distribution

                critic_value = T.squeeze(critic_value)
                new_probs = actions_distribution.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() #why negative?
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2 #using MSE mean squared error
                critic_loss = critic_loss.mean()

                entropy_loss = T.mean(new_m.entropy())
                total_loss = actor_loss + self.critic_discount*critic_loss - (self.beta * entropy_loss)

                ## Calculate the critic scaling factor dynamically.
                actor_loss = actor_loss.detach().cpu().numpy().astype('float32')
                actor_loss = round(float(actor_loss),4)
                critic_loss = critic_loss.detach().cpu().numpy().astype('float32')
                critic_loss = round(float(critic_loss),4)
                perfect_reduction = round(abs(actor_loss)/abs(critic_loss),4)
                self.critic_discount_targets.append(perfect_reduction)
                
                """Troubleshooting
                adjusted_critic = round(self.critic_discount*critic_loss,4)
                adjusted_delta = round(abs(actor_loss)-abs(self.critic_discount*critic_loss),4)
                print("Loss Check....", end="")
                print(f"Actor:{actor_loss}; Critic:{critic_loss};", end = "")
                print(f"Adjusted Critic:{adjusted_critic};Adjusted Delta:{adjusted_delta}", end = "")
                print(f"Perfect Reduction = {actor_loss/critic_loss}")
                """

                self.model.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) #performs gradient clipping. It is used to mitigate the problem of exploding gradients,
                self.model.optimizer.step()

                """
                total_loss = actor_loss + 0.5*critic_loss
                self.model.optimizer.zero_grad()
                total_loss.backward()
                self.model.optimizer.step()
                """
        ## Calculate the critic scaling dynamically.
        self.critic_discount = np.mean(self.critic_discount_targets)
        self.critic_discount_targets = []
        #print(f"Final Critic Discount:{self.critic_discount}")

        self.memory.clear_memory()               

    def image_to_tensor(self, image):
        #4 frames are staggered across time. 30cycles/second = 30 frame buffer of 1 second. We select frames 
        # 0,15,22,29; with the newest frame being on the end(24).
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor.astype(np.float32)
        self.complex_buffer.append(image_tensor)
        
        #"""
        #the following protocol handles the 4frame "movement" array.
        if len(self.complex_buffer) < 30: #Deque hasnt filled
            self.state_array = np.concatenate((image_tensor, image_tensor, image_tensor, image_tensor), axis=0)
            #self.state_array = np.expand_dims(self.state_array, axis=0)
        else:
            self.state_array = np.concatenate((self.complex_buffer[0], self.complex_buffer[15], 
                                                self.complex_buffer[22], self.complex_buffer[29]), axis=0)
            #self.state_array = np.concatenate((self.state_array[1:, :, :], image_tensor), axis=0)
        #"""
        return self.state_array 

if __name__ == "__main__":
    parameters = {
        ## Hyperparameters:
        'learning_rate':1e-6, #5e-6, 9e-6
        'gamma': 0.975, #0.95; Discount Factor
        'tau': 0.975, #0.8, 0.65; Scaling parameter for GAE
        'beta':0.005, #0.01; Entropy coefficient; induces more random exploration
        'epsilon': 0.1, #parameter for Clipped Surrogate Objective
        'epochs': 5, #how many times do you want to learn from the data
        'number_of_iterations': 20000000,
        'game_count_limit': 100, #how many games do we wish to run this for.
        'batch_size': 200, #5 
        'num_local_steps': 800, #Used for dataset learning. total learning steps at learn time.
    }

    #merge = NNMerge(parameters) #give it the whole dict.
    #merge.merge_models()
    agent = Agent(parameters)
