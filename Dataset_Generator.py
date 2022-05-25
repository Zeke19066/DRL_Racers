"""
How it works:

the dataset is generated at 6ps of gameplay, where each game is a folder and the move at the moment is saved in the file name.
Folder name: Timestamp
File name: Chorono Number, action.
The images are in the same format that the agent will be seeing when testing the performance in live gameplay.
"""
import os
import time
import cv2
from PIL import Image
from datetime import datetime
import numpy as np

from decorators import function_timer
import Racers_Wrapper
import keyboard  # For Keylogging


class Dataset_Generator():

    def __init__(self):
        self.screen = Racers_Wrapper.ScreenGrab()
        self.controller = Racers_Wrapper.Actions()
        #self.controller.first_Run()
        self.master_path = "Dataset"
        self.home_dir = os.getcwd()

        self.timing_target = 33 # target fps capture rate; must be less than unthrottled.
        self.timing_offset = 0 #this will be set in set_framerate()
        self.unthrottled_fps = 33.792917

        self.set_framerate()

    def set_framerate(self):

        #used to determine unthrottled fps.
        @function_timer
        def screengrab_subfunction():
            for i in range(1000):
                image_big = self.screen.quick_Grab()
                image = self.screen.resize(image_big, False)
        
                #cv2.imshow("cv2screen", image)
                #cv2.waitKey(10)
                file_dir_name = "test_delete.png"
                status = cv2.imwrite(file_dir_name, image)
                time.sleep(0.0000000000000001) #account for sleep function call time overhead.
        #screengrab_subfunction()

        self.timing_offset = (1/self.timing_target)-(1/self.unthrottled_fps)

    def Generator(self):
        #subfunction to find out what key we pressed.
        def key_watcher():
            #action_set = [0'Forward', 1'Powerup', 2'Reverse', 3'Left', 4'Right']
            while True:  # making a loop
                try:  # used try so that if user pressed other than the given key error will not be shown
                    if keyboard.is_pressed('w'):  # Forward
                        if not keyboard.is_pressed('a') and not keyboard.is_pressed('d'):
                            master_action = 0
                            break
                    if keyboard.is_pressed('q'):  # Fire Powerup
                        master_action = 1
                        break
                    if keyboard.is_pressed('a'):  # Left
                        master_action = 3 
                        break
                    if keyboard.is_pressed('d'):  # Right
                        master_action = 4 
                        break
                    if keyboard.is_pressed('s'):  # Reverse
                        master_action = 2 
                        break
                except:
                    pass
            return master_action

        frame_counter = 0
        #self.controller.Reset() # Reset the game state for a clean start.
        race_over_bool = False
        os.chdir(self.home_dir) #make sure we're in the main folder

        #Create the game folder 
        folder_timestamp_raw = datetime.now()
        folder_timestamp = folder_timestamp_raw.strftime("%m.%d.%Y, %H.%M.%S")
        folder_name = self.master_path+"//"+str(folder_timestamp)
        os.mkdir(folder_name)
        os.chdir(folder_name)

        #this checks for GO! screen to trigger loop
        print("Waiting for Green Light......", end="")
        while 1: #Check to see when its time to go.
            img = self.screen.quick_Grab()
            go_crop = img[154:316, 363:511]# Wrong Way Coords in [Y1:Y2, X1:X2] format
            go_bool = self.screen.go_scan(go_crop)
            if go_bool:
                print('GO!')
                break


        # main cycle loop
        print('Entering Gamecapture Loop')
        while 1: 

            #get screenshot
            image_big = self.screen.quick_Grab() # Get the next screenshot
            image_small = self.screen.resize(image_big, False)
            
            #cv2.imshow("cv2screen", image)
            #cv2.waitKey(10)

            #get keypress
            keypress = key_watcher()
            file_dir_name = str(frame_counter) + "," + str(keypress) + ".png"
            status = cv2.imwrite(file_dir_name, image_small)
            if frame_counter%100==0:
                print(f"Count:{frame_counter} Image:{status}")
            frame_counter += 1

            #check to see if the race is over.
            if frame_counter > 1000:
                finish_image = image_big[165:252, 340:528]
                race_over_bool = self.screen.race_over(finish_image)
                if race_over_bool:
                    new_time = datetime.now()
                    time_delta = (new_time - folder_timestamp_raw).total_seconds()
                    cycles_per_second = round(frame_counter/time_delta,2)
                    print(f"Race Over! SaveRate:{cycles_per_second}frames/sec  Resetting....")
                    return
            
            time.sleep(self.timing_offset) #gets us into the framerate we want.
            
if __name__ == "__main__":
    dataset = Dataset_Generator()
    while 1:
        dataset.Generator()
        pass
