import os
import numpy as np
import cv2

"""
Test script for loading dataset examples and extracting
meta information from the file name.

Naming Convention: Index,Action.png
"""

class Image_Loader():

    def __init__(self):
        pass
    #subfunction for natural sorting of filename list.
    def natural_sorter(self, data):
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    #a subfunction for extracting the 
    def meta_extractor(self, filename_list):
        action_list = []
        for filename in filename_list:
            out = []
            demarcation_bool = False
            for c in filename:
                
                #exclude "".png"
                if c ==".":
                    demarcation_bool = False
                if demarcation_bool:
                    out.append(c)
                if c ==",":
                    demarcation_bool = True
            answer = "".join(out)
            action_list.append(answer)
        return action_list

    ##This function shows dataset images sequentially
    def viewer(self):
        #Chose the random folder and load the first image. 
        os.chdir("..")#move up one directory level.
        os.chdir("Dataset")
        folder_list = os.listdir()
        selection = np.random.randint(len(folder_list))
        subfolder = folder_list[selection]
        #now we're in the folder loading the file list
        os.chdir(subfolder)
        file_list = os.listdir()
        file_list = self.natural_sorter(file_list)
        img_index = 0
        #make a seperate list of actions from filename
        action_list = self.meta_extractor(file_list)
        print(action_list)
        #print(file_list)
        #"""
        while img_index<len(file_list):
            current_image = cv2.imread(file_list[img_index], cv2.IMREAD_GRAYSCALE)
            cv2.imshow("cv2screen", current_image)
            cv2.waitKey(1)

            img_index+=1
        #"""

    ##This function analyzes the dataset 
    def analyzer(self):
        #Get all the file names for all of the dataset folders.
        os.chdir("..")#move up one directory level.
        os.chdir("Dataset")
        folder_list = os.listdir()
        all_files_list = []
        for folder in folder_list:
            os.chdir(str(folder))
            file_list = os.listdir()
            file_list = self.natural_sorter(file_list)
            all_files_list = all_files_list + file_list
            os.chdir("..")#move up one directory level.

        out_dict = {0:0, 1:0, 2:0, 3:0, 4:0}
        action_list = self.meta_extractor(all_files_list)
        for action in action_list:
            out_dict[int(action)] += 1
        total_actions = out_dict[0]+out_dict[1]+out_dict[2]+out_dict[3]+out_dict[4]
        for action in out_dict:
            out_dict[action] = round(out_dict[action]/total_actions*100,2)
        print(out_dict)

if __name__ == "__main__":
    img_loader = Image_Loader()
    img_loader.analyzer()