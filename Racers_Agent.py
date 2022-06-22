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
from collections import deque
import torch as T
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F

#The A2C model structure
class CNNActor(nn.Module):

    def __init__(self, num_inputs, num_actions, alpha, 
                 file_name='pretrained_model/current_model_actor.pth'):
        super(CNNActor, self).__init__()
        
        self.checkpoint_file = file_name
        self.number_of_actions = num_actions
        self.flat_size = 9216

        #in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.linear = nn.Linear(self.flat_size, 512)
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
        #This needs to be calculated manually for input img size change.
        #print(f'***Flattened Shape: {out.shape}***')
        return self.actor_linear(out)

    def save_checkpoint(self, 
                        file_name='pretrained_model/current_model_actor.pth'):
        os.chdir(self.home_dir) #make sure we're in the main folder
        T.save(self.state_dict(), file_name)

    def load_checkpoint(self,
                        file_name='pretrained_model/current_model_actor.pth'):
        os.chdir(self.home_dir) #make sure we're in the main folder
        self.load_state_dict(T.load(file_name)) #strict=False

class CNNCritic(nn.Module):

    def __init__(self, num_inputs, num_actions, alpha, 
                 file_name='pretrained_model/current_model_critic.pth'):
        super(CNNCritic, self).__init__()
        
        self.checkpoint_file = file_name
        self.flat_size = 9216

         #in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.linear = nn.Linear(self.flat_size, 512)
        self.critic_linear = nn.Linear(512, 1)

        self.home_dir = os.getcwd()

        file_exists = exists(self.checkpoint_file)
        if file_exists:
            self.load_checkpoint(self.checkpoint_file)
        else:
            self._initialize_weights()
            self.save_checkpoint(self.checkpoint_file)

         # define Adam optimizer;
        self.optimizer = optim.Adam(self.parameters(), alpha)
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
        #This needs to be calculated manually for input img size change.
        #print(f'***Flattened Shape: {out.shape}***')
        return self.critic_linear(out)

    def save_checkpoint(self,
                        file_name='pretrained_model/current_model_critic.pth'):
        os.chdir(self.home_dir) #make sure we're in the main folder
        T.save(self.state_dict(), file_name)

    def load_checkpoint(self, 
                        file_name='pretrained_model/current_model_critic.pth'):
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

        #only one element tensors can be converted to Python scalars
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

class CNNMerge:

    def __init__(self, parameters_dict, num_actions):
        self.parameters_dict = parameters_dict
        self.num_actions = num_actions

    def merge_models(self):
        self.parent_1 = CNNActor(4, self.num_actions,
                        self.parameters_dict["actor_learning_rate"],
                        file_name='pretrained_model/Merge_Folder/parent1.pth')
        self.parent_2 = CNNActor(4, self.num_actions, 
                        self.parameters_dict["actor_learning_rate"],
                        file_name='pretrained_model/Merge_Folder/parent2.pth')
        self.parent_3 = CNNActor(4, self.num_actions,
                        self.parameters_dict["actor_learning_rate"],
                        file_name='pretrained_model/Merge_Folder/parent3.pth')
        self.parent_4 = CNNActor(4, self.num_actions,
                        self.parameters_dict["actor_learning_rate"],
                        file_name='pretrained_model/Merge_Folder/parent4.pth')
        self.parent_5 = CNNActor(4, self.num_actions,
                        self.parameters_dict["actor_learning_rate"],
                        file_name='pretrained_model/Merge_Folder/parent5.pth')
        self.merge_out = CNNActor(4, self.num_actions,
                         self.parameters_dict["actor_learning_rate"],
                         file_name='pretrained_model/current_model_actor.pth')
    
        #convert tensors to numpy arrays
        np_parent_1 = self.model2numpy(self.parent_1)
        np_parent_2 = self.model2numpy(self.parent_2)
        np_parent_3 = self.model2numpy(self.parent_3)
        np_parent_4 = self.model2numpy(self.parent_4)
        np_parent_5 = self.model2numpy(self.parent_5)
        np_merge_output = []

        for n in range(len(np_parent_1)):
            average_array = (np_parent_1[n]+np_parent_2[n]
                             +np_parent_3[n]+np_parent_4[n]
                             +np_parent_5[n])/5
            np_merge_output.append(average_array)

        #now convert nump array output to tensors in model.
        self.numpy2model(np_merge_output)
        self.merge_save()

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
        np_actor_linear_bias = model.actor_linear.bias.detach().cpu().numpy().astype('float32')
        np_model.append(np_actor_linear_bias)

        return np_model

    def numpy2model(self, np_model):

        ##First do the weights 
        tensor_conv1_weights = T.tensor(np_model[0], dtype=T.float)
        self.merge_out.conv1.weight = nn.Parameter(
                                    data=tensor_conv1_weights,
                                    requires_grad=True)
        tensor_conv2_weights = T.tensor(np_model[1], dtype=T.float)
        self.merge_out.conv2.weight = nn.Parameter(
                                    data=tensor_conv2_weights,
                                    requires_grad=True)
        tensor_conv3_weights = T.tensor(np_model[2], dtype=T.float)
        self.merge_out.conv3.weight = nn.Parameter(
                                    data=tensor_conv3_weights,
                                    requires_grad=True)
        tensor_linear_weights =  T.tensor(np_model[3], dtype=T.float)
        self.merge_out.linear.weight = nn.Parameter(
                                     data=tensor_linear_weights,
                                     requires_grad=True)
        tensor_actor_linear_weights = T.tensor(np_model[4], dtype=T.float)
        self.merge_out.actor_linear.weight = nn.Parameter(
                                           data=tensor_actor_linear_weights,
                                           requires_grad=True)
        
        #now the biases
        tensor_conv1_bias = T.tensor(np_model[5], dtype=T.float)
        self.merge_out.conv1.bias = nn.Parameter(
                                  data=tensor_conv1_bias,
                                  requires_grad=True)
        tensor_conv2_bias = T.tensor(np_model[6], dtype=T.float)
        self.merge_out.conv2.bias = nn.Parameter(
                                  data=tensor_conv2_bias,
                                  requires_grad=True)
        tensor_conv3_bias = T.tensor(np_model[7], dtype=T.float)
        self.merge_out.conv3.bias = nn.Parameter(
                                  data=tensor_conv3_bias,
                                  requires_grad=True)
        tensor_linear_bias =  T.tensor(np_model[8], dtype=T.float)
        self.merge_out.linear.bias = nn.Parameter(
                                   data=tensor_linear_bias,
                                   requires_grad=True)
        tensor_actor_linear_bias = T.tensor(np_model[9], dtype=T.float)
        self.merge_out.actor_linear.bias = nn.Parameter(
                                         data=tensor_actor_linear_bias,
                                         requires_grad=True)
        
    def merge_save(self):
        print('... Merge complete: saving model ...')
        name='pretrained_model/current_model_actor.pth'
        self.merge_out.save_checkpoint(file_name=name)

class Agent:
    def __init__(self, parameters_dict, num_actions):

        # gamma=0.99, alpha=0.0003, gae_lambda=0.95,
        #   policy_clip=0.2, batch_size=64, n_epochs=10

        self.gamma = parameters_dict["gamma"]
        self.policy_clip = parameters_dict["epsilon"]
        self.n_epochs = parameters_dict["epochs"]
        self.gae_lambda = parameters_dict["tau"]
        self.beta = parameters_dict["beta"]
        self.parameters_dict = parameters_dict
        que_len = round(self.n_epochs*25)
        self.actor_loss_que = deque(maxlen=que_len)
        self.critic_loss_que = deque(maxlen=que_len)
        self.entropy_loss_que = deque(maxlen=que_len)

        self.initial_state_bool = True
        self.state_tensor = []
        self.state_array = []
        self.loss_print = ""
        #we average 30 cycles/second;yields 1 sec ref window to sample.
        self.complex_buffer = deque(maxlen=30) 

        self.actor_model= CNNActor(4, num_actions,
                                     parameters_dict["actor_learning_rate"])
        self.critic_model= CNNCritic(4, num_actions,
                                     parameters_dict["critic_learning_rate"])
        self.memory = PPOMemory(parameters_dict["mini_batch_size"])
        self.merge = CNNMerge(parameters_dict, num_actions) #For merging NN

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving model ...')
        self.actor_model.save_checkpoint()
        self.critic_model.save_checkpoint()

    def load_models(self):
        print('... loading model ...')
        self.actor_model.load_checkpoint()
        self.critic_model.load_checkpoint()

    def merge_models(self):
        self.merge.merge_models()

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).to(self.actor_model.device)
        state = state.squeeze(0)
        state = state.unsqueeze(0)

        action_space, value = self.actor_model(state), self.critic_model(state)
        actions_dist = F.softmax(action_space, dim=1)
        actions_dist = Categorical(actions_dist)
        action = actions_dist.sample()

        probs = T.squeeze(actions_dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value, action_space

    def learn(self):
        session_actor_loss = []
        session_critic_loss = []
        session_entropy_loss = []


        #subfunction for so len(data)%batch size = 0; evenly trims for zero remainder.
        #even though the mask is generated each time, its set to len&divisor. Same len&divisor = same mask every time.
        def array_adjust(array, divisor):
            input_len = len(array)
            remainder = input_len%divisor
            #if there is no remainder, we return the original array.
            if remainder != 0:#create a mask of what indexies to remove.
                mask_rate = np.floor(input_len/remainder)
                advantage = np.zeros(remainder, dtype=np.int32)
                for i, val in enumerate(advantage):
                    advantage[i] = i*mask_rate
                #now apply the mask
                for val in advantage:
                    array.pop(val)
            return array
        
        self.memory.batch_size = int(len(self.memory.states) /5)
        #all arrays have same length
        self.memory.states = array_adjust(self.memory.states,
                                          self.memory.batch_size)
        self.memory.actions = array_adjust(self.memory.actions,
                                           self.memory.batch_size)
        self.memory.probs = array_adjust(self.memory.probs,
                                         self.memory.batch_size)
        self.memory.vals = array_adjust(self.memory.vals,
                                        self.memory.batch_size)
        self.memory.rewards = array_adjust(self.memory.rewards,
                                           self.memory.batch_size)
        self.memory.dones = array_adjust(self.memory.dones,
                                         self.memory.batch_size)

        for epoch in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr #Array of critic values
            advantage = np.zeros(len(reward_arr), dtype=np.float32) #initialize advantage array as all zeros.

            for t in range(len(reward_arr)-1): #t = timestep; for every timestep in the reward array.
                discount = 1
                a_t = 0 # advantage at timestep initialized as zero.
                for k in range(t, len(reward_arr)-1): #k= timesteps to the end; for every timestep in the future.
                    a_t += (discount*(reward_arr[k] + self.gamma*values[k+1]
                            *(1-int(dones_arr[k])) - values[k]))
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor_model.device)
            values = T.tensor(values).to(self.critic_model.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor_model.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor_model.device)
                actions = T.tensor(action_arr[batch]).to(self.actor_model.device)

                actions_dist = self.actor_model(states)
                critic_val = self.critic_model(states)
                actions_dist = F.softmax(actions_dist, dim=1)
                actions_dist = Categorical(actions_dist)
                new_m = actions_dist

                critic_val = T.squeeze(critic_val)
                new_probs = actions_dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clip_probs = (T.clamp(prob_ratio,
                                       1-self.policy_clip,
                                       1+self.policy_clip)
                                       *advantage[batch])
                
                ## Calculate the Actor Loss.
                entropy_loss = T.mean(new_m.entropy())
                entropy_loss = (self.beta * entropy_loss) #subtract this
                actor_loss = -T.min(weighted_probs,
                                    weighted_clip_probs).mean()
                #save version w/o entropy credit
                actor_loss_copy = T.clone(actor_loss).to(self.critic_model.device)

                #we want a net credit towards zero
                if actor_loss < 0:
                    actor_loss = actor_loss+entropy_loss
                if actor_loss >= 0:
                    actor_loss = actor_loss-entropy_loss

                ## Calculate the Critic Loss.
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_val)**2
                critic_loss = critic_loss.mean()

                ## Update the network; gradient descent
                self.actor_model.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.critic_model.optimizer.zero_grad()
                critic_loss.backward()

                #performs gradient clipping.
                #It is used to mitigate the problem of exploding gradients.
                nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.5)
                
                self.actor_model.optimizer.step()
                self.critic_model.optimizer.step()

                #take the entropy back out for better insight.
                actor_loss = actor_loss_copy
                actor_loss = actor_loss.detach().cpu().numpy().astype('float32')
                actor_loss = round(float(actor_loss),4)
                critic_loss = critic_loss.detach().cpu().numpy().astype('float32')
                critic_loss = round(float(critic_loss),4)
                entropy_loss = entropy_loss.detach().cpu().numpy().astype('float32')
                entropy_loss = round(float(entropy_loss),4)
                session_actor_loss.append(actor_loss)
                self.actor_loss_que.append(actor_loss)
                session_critic_loss.append(critic_loss)
                self.critic_loss_que.append(critic_loss)
                session_entropy_loss.append(entropy_loss)
                self.entropy_loss_que.append(entropy_loss)
            print(f"...{epoch+1}/{self.n_epochs}", end="")
        

        #we're generating the string so for the GUI as well.
        print("...Epochs Complete")
        print_1 = "Loss Check(Session/Avg 25)..."
        val1 = round(np.mean(session_actor_loss),4)
        val2 = round(np.mean(self.actor_loss_que),4)
        print_2 = f"Actor: {val1}/{val2}; "
        val1 = round(np.mean(session_critic_loss))
        val2 = round(np.mean(self.critic_loss_que))
        print_3 = f"Critic: {val1}/{val2}; "
        val1 = round(np.mean(session_entropy_loss),4)
        val2 = round(np.mean(self.entropy_loss_que),4)
        print_4 = f"Entropy: {val1}/{val2}"
        self.loss_print = print_1 + print_2 + print_3 + print_4
        print(self.loss_print)
        self.memory.clear_memory()

    def image_to_tensor(self, image):
        #4 frames are staggered across time. 30cycles/second = 30 frame buffer of 1 second. We select frames 
        # 0,15,22,29; with the newest frame being on the end(24).
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor.astype(np.float32)
        self.complex_buffer.append(image_tensor)
        
        #the following protocol handles the 4frame "movement" array.
        if len(self.complex_buffer) < 30: #Deque hasnt filled
            self.state_array = np.concatenate((image_tensor, image_tensor, 
                                               image_tensor, image_tensor),
                                               axis=0)
        else:
            self.state_array = np.concatenate((self.complex_buffer[0], 
                                               self.complex_buffer[15], 
                                               self.complex_buffer[22],
                                               self.complex_buffer[29]),
                                               axis=0)

        return self.state_array 

if __name__ == "__main__":
    parameters = {
        ## Hyperparameters:
        'actor_learning_rate':1e-6, #5e-6, 9e-6
        'gamma': 0.975, #0.95; Discount Factor
        'tau': 0.975, #0.8, 0.65; Scaling parameter for GAE
        'beta':0.005, #0.01; Entropy coefficient; induces more random exploration
        'epsilon': 0.1, #parameter for Clipped Surrogate Objective
        'epochs': 5, #how many times do you want to learn from the data
        'number_of_iterations': 20000000,
        'game_count_limit': 100, #how many games do we wish to run this for.
        'mini_batch_size': 200, #5 
        'num_local_steps': 800, #Used for dataset learning. total learning steps at learn time.
    }

    merge = CNNMerge(parameters,5) #give it the whole dict.
    merge.merge_models()
    #agent = Agent(parameters, 10)
