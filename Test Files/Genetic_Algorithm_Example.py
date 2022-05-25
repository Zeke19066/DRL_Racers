import torch
import pygad.torchga
import pygad
import numpy as np

import cv2
import os

import torch.nn as nn

class Genetic_Algorithm(nn.Module):

    def __init__(self):
        super(Genetic_Algorithm, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.initial_state_bool = True
        self.state_tensor = [] #initialize empty.
        self.state_array = []

        self.home_dir = os.chdir("..")
        self.home_dir = os.getcwd()

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
            return self.state_array 


    #loading dataset images.
    def data_loader(self):

        #subfunction for natural sorting of file names.
        import re
        def natural_sorter(data):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
            return sorted(data, key=alphanum_key)

        #a subfunction for extracting the human actions from filenames
        def meta_extractor(filename_list):
            action_list = []
            for filename in filename_list:
                out = []
                demarcation_bool = False
                for c in filename:
                    #exclude ".png"
                    if c ==".":
                        demarcation_bool = False
                    if demarcation_bool:
                        out.append(c)
                    if c ==",":
                        demarcation_bool = True
                answer = "".join(out)
                action_list.append(int(answer))
            return action_list

        print("Initializing Dataset....", end="")
        #Chose the random folder and load the first image. 
        os.chdir(self.home_dir)
        os.chdir("Dataset")
        folder_list = os.listdir()
        selection = np.random.randint(len(folder_list))
        self.subfolder = folder_list[selection]
        #now we're in the folder loading the file list
        os.chdir(self.subfolder)
        self.file_list = os.listdir()
        self.file_list = natural_sorter(self.file_list)
        self.action_list = meta_extractor(self.file_list)#make a seperate list of actions from filename

        #preload all the images:
        self.frame_buffer = []
        square_size = 126
        for file_name in self.file_list:
            sub_frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_GRAYSCALE
            sub_frame = np.reshape(sub_frame, (square_size, square_size, 1))
            sub_frame_tensor = self.image_to_tensor(sub_frame)
            self.frame_buffer.append(sub_frame_tensor)

        print("Done -Length:",len(self.action_list))

        return self.frame_buffer, self.action_list

    def main_body(self):

        def fitness_func(solution, sol_idx):
            predictions = pygad.torchga.predict(model=self.model, 
                                                solution=solution, 
                                                data=self.data_inputs)

            solution_fitness = 1.0 / (self.loss_function(predictions, self.data_outputs).detach().np() + 0.00000001)

            return solution_fitness

        def callback_generation(ga_instance):
            print(f"Generation = {ga_instance.generations_completed}", end = "        ")
            print(f"Fitness = {ga_instance.best_solution()[1]}")

        def model_init():# Build the PyTorch model.

            number_of_actions = 5 # How many output acitons?
            flat_size = 9216

            conv1 = nn.Conv2d(4, 32, 8, 4) #in_channels, out_channels, kernel_size, stride, padding
            conv2 = nn.Conv2d(32, 64, 4, 2)
            conv3 = nn.Conv2d(64, 64, 3, 1)
            relu3 = nn.ReLU(inplace=True)
            fc4 = nn.Linear(flat_size, 512)
            relu4 = nn.ReLU(inplace=True)
            fc5 = nn.Linear(512, number_of_actions)
            output_layer = nn.Softmax(1)

            model = nn.Sequential(conv1,conv2,conv3,relu3,fc4,relu4,fc5,output_layer)

            """ Original Version
            input_layer = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=7)
            relu_layer1 = torch.nn.ReLU()
            max_pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=5)

            conv_layer2 = torch.nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3)
            relu_layer2 = torch.nn.ReLU()

            flatten_layer1 = torch.nn.Flatten()
            # The value 768 is pre-computed by tracing the sizes of the layers' outputs.
            dense_layer1 = torch.nn.Linear(in_features=768, out_features=15)
            relu_layer3 = torch.nn.ReLU()

            dense_layer2 = torch.nn.Linear(in_features=15, out_features=4)
            output_layer = torch.nn.Softmax(1)

            model = torch.nn.Sequential(input_layer,
                                        relu_layer1,
                                        max_pool1,
                                        conv_layer2,
                                        relu_layer2,
                                        flatten_layer1,
                                        dense_layer1,
                                        relu_layer3,
                                        dense_layer2,
                                        output_layer)
            """
            
            return model

        self.model = model_init()
        # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
        self.torch_ga = pygad.torchga.TorchGA(model=self.model,
                                        num_solutions=10)

        self.loss_function = torch.nn.CrossEntropyLoss()

        ## Data inputs; Here the images are loaded such that data_inputs = [[img1],[img2],[img3], etc.]
        #data_inputs = torch.from_numpy(numpy.load("dataset_inputs.npy")).float() #original, condensed version
        data_inputs, data_outputs = self.data_loader()
        data_inputs, data_outputs = np.array(data_inputs), np.array(data_outputs)
        print(f"inputs length:{len(data_inputs)}", end="...........")
        """
        for img in data_inputs:
            cv2.imshow("cv2screen", img)
            cv2.waitKey(1)
        """
        self.data_inputs = torch.from_numpy(data_inputs).float()
        #self.data_inputs = data_inputs.reshape((data_inputs.shape[0], data_inputs.shape[3], data_inputs.shape[1], data_inputs.shape[2]))


        ## Data outputs
        #self.data_outputs = torch.from_numpy(numpy.load("dataset_outputs.npy")).long() #original, condensed version
        print(f"outputs length:{len(data_inputs)}")
        print(data_outputs)
        self.data_outputs = torch.from_numpy(data_outputs).long()

        # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
        num_generations = 200 # Number of generations.
        num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
        initial_population = self.torch_ga.population_weights # Initial population of network weights.

        # Create an instance of the pygad.GA class
        ga_instance = pygad.GA(num_generations=num_generations, 
                            num_parents_mating=num_parents_mating, 
                            initial_population=initial_population,
                            fitness_func=fitness_func,
                            on_generation=callback_generation)

        # Start the genetic algorithm evolution.
        ga_instance.run()

        # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
        ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        predictions = pygad.torchga.predict(model=self.model, 
                                            solution=solution, 
                                            data=self.data_inputs)
        # print("Predictions : \n", predictions)

        # Calculate the crossentropy for the trained model.
        print("Crossentropy : ", self.loss_function(predictions, self.data_outputs).detach().numpy())

        # Calculate the classification accuracy for the trained model.
        accuracy = torch.true_divide(torch.sum(torch.max(predictions, axis=1).indices == self.data_outputs), len(self.data_outputs))
        print("Accuracy : ", accuracy.detach().numpy())

if __name__ == "__main__":
    algo = Genetic_Algorithm()
    algo.main_body()