"""
How it works:
Everyone uses the same conv layers for feature extraction, and the split between agent and critic only happens in the last layer.
The initialization weights look a little wonky as well, but it might just be more modern.

This is called from train()
"""
from os.path import exists
import os
import numpy as np
import torch as T
import torch.optim as optim
from torch.distributions.categorical import Categorical


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
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.linear = nn.Linear(self.flat_size, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        #self._initialize_weights()

        file_exists = exists(self.checkpoint_file)
        if file_exists:
            self.load_checkpoint()
        else:
            self._initialize_weights()
            self.save_checkpoint()

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
        T.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name='pretrained_model/current_model_seed.pth'):
        self.load_state_dict(T.load(file_name))

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

class Agent:
    def __init__(self, parameters_dict):

        # gamma=0.99, alpha=0.0003, gae_lambda=0.95,
        #   policy_clip=0.2, batch_size=64, n_epochs=10

        self.gamma = parameters_dict["gamma"]
        self.policy_clip = parameters_dict["epsilon"]
        self.n_epochs = parameters_dict["epochs"]
        self.gae_lambda = parameters_dict["tau"]
        self.beta = parameters_dict["beta"]

        self.initial_state_bool = True
        self.state_tensor = [] #initialize empty.
        self.state_array = []

        self.model = PPONetwork(4, 5, parameters_dict["learning_rate"])
        self.memory = PPOMemory(parameters_dict["batch_size"])
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, file_name='pretrained_model/current_model_seed.pth'):
        print('... saving model ...')
        self.model.save_checkpoint(file_name)

    def load_models(self, file_name='pretrained_model/current_model_seed.pth'):
        print('... loading model ...')
        self.model.load_checkpoint(file_name)

    def choose_action(self, state):
        #state = T.tensor([observation], dtype=T.float).to(self.model.device)
        state = T.tensor(state, dtype=T.float).to(self.model.device)
        state = state.squeeze(0)
        state = state.unsqueeze(0)
        #if T.cuda.is_available():  # put on GPU if CUDA is available
        #    state = state.cuda()

        action_space, value = self.model(state)
        dist = F.softmax(action_space, dim=1)
        dist = Categorical(dist)

        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value, action_space

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
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

                dist, critic_value = self.model(states)
                dist = F.softmax(dist, dim=1)
                dist = Categorical(dist)
                new_m = dist

                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                entropy_loss = T.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - self.beta * entropy_loss
                #total_loss = actor_loss + 0.5*critic_loss - self.beta * entropy_loss
                self.model.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.model.optimizer.step()

                """
                total_loss = actor_loss + 0.5*critic_loss
                self.model.optimizer.zero_grad()
                total_loss.backward()
                self.model.optimizer.step()
                """

        self.memory.clear_memory()               

    def image_to_tensor(self, image):
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor.astype(np.float32)
        
        #"""
        #the following protocol handles the 4frame "movement" array.
        if self.initial_state_bool:
            self.initial_state_bool = False
            self.state_array = np.concatenate((image_tensor, image_tensor, image_tensor, image_tensor), axis=0)
            #self.state_array = np.expand_dims(self.state_array, axis=0)
        elif not self.initial_state_bool:
            #self.state_array = np.concatenate((self.state_array.squeeze(0)[1:, :, :], image_tensor), axis=0)
            #self.state_array = np.expand_dims(self.state_array, axis=0)
            self.state_array = np.concatenate((self.state_array[1:, :, :], image_tensor), axis=0)
        #"""
        """
        image_tensor = T.from_numpy(image_tensor) # Creates a Tensor from a numpy.ndarray (Nth Dimension Array).
        if self.initial_state_bool:
            self.initial_state_bool = False
            self.state_tensor = T.cat((image_tensor, image_tensor, image_tensor, image_tensor))#.unsqueeze(0)
        elif not self.initial_state_bool:
            self.state_tensor = T.cat((self.state_tensor.squeeze(0)[1:, :, :], image_tensor))#.unsqueeze(0)

        state_np_array = self.state_tensor.cpu().detach().numpy() #torch.Size([1, 4, 126, 126])
        print(state_np_array.shape)
        return state_np_array #(4, 126, 126)
        """

        return self.state_array #[20, 1, 4, 126, 126] instead
