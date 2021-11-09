import random
import os
import time
import numpy as np
import json 
from collections import deque
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from decorators import function_timer



# Neural Network Parameters & Hyperparameters
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 5 # How many output acitons?
        self.flat_size = 10816 #size of the flattened input image (out.shape to get size)

        self.conv1 = nn.Conv2d(4, 32, 8, 4) #in_channels, out_channels, kernel_size, stride, padding
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(self.flat_size, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    # Forwardpass Method
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        #print(f'Flattened Shape: {out.shape}') #This needs to be calculater as this image shape needs to feed into next layer.
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

# Intialization weights
def initial_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)



def main():
    model = NeuralNetwork()
    file_letters = ['A','B','C','D','E']
    loaded_seed = str(file_letters[0])
    model_name = 'pretrained_model/current_model_seed_EVO_'+ str(file_letters[0]) +'.pth'
    model_dict = torch.load(model_name)
    model.load_state_dict(model_dict)
    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(pytorch_total_params)
    #print(model.parameters())

    print(type(model_dict))
    
    for key, item in model_dict.items():
        print("********************************************************")
        print(key,item)
        """
        for subkey, subitem in enumerate(item):
            if type(subitem) == torch.Tensor:
                size = subitem.size()
                #print(size, str(size))
                if str(size) == "torch.Size([])":
                    print(subitem.size())
                else:
                    for subsubitem in enumerate(subitem):
                        for super_subitem in subsubitem:
                            for final_item in super_subitem:
                                print(final_item)
        """

if __name__ == "__main__":
    main() #Local initialization won't use the 
