import time
import numpy as np
import math
import cv2
import os
import win32gui, win32ui, win32con, win32api # for quick_Grab
from Custom_Keymapper import Custom_Keymapper as key_mapper #Custom C++ keymap library (Fast)
from decorators import function_timer

import pyautogui
import pydirectinput #this module requires pyautogui to run
import keyboard  # For Keylogging

import matplotlib.pyplot as plot

'''
This file handles the screenshot & image processing, and the Keystroke interactions with the Game-Interface.

Notes:
The gamescreen offset is (52,104).
self.action_set = [0'Forward', 1'Powerup', 2'Reverse', 3'Left', 4'Right']
'''

# Where screencaptures are executed
class ScreenGrab():

    def __init__(self):
        self.x_offset = 104-1
        self.y_offset = 52-1
        self.show_me_stale = 0
        self.home_dir = os.getcwd()

        

    #Quick Screenshot
    def quick_Grab(self, region=None):
        hwin = win32gui.GetDesktopWindow()
        if region:
            left, top, width, height = 104, 52, (1922-104 +1), (1472-52  +1)

        else:
            left, top, width, height = 104, 52, (1922-104 +1), (1472-52  +1)

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        #img = np.fromstring(signedIntsArray, dtype='uint8')
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height,width,4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) #retruns the RGB Image
        return img

    def resize(self, img, wrong_way_bool=False):
        xy = int(504/4)
        img = np.array(img)
        img = cv2.resize(img, (xy, xy))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img) #equalize the image for greater contrast.
        if wrong_way_bool: #invert image if we're going the wrong way.
            img = ~img
        img = np.reshape(img, (xy, xy, 1))
        return img
        
    #loading dataset images.
    def data_loader(self, first_run=False, reset=False):
        gameover_bool = False
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
                action_list.append(answer)
            return action_list

        if reset:
            self.img_index=0

        if first_run:
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
            for file_name in self.file_list:
                sub_frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_GRAYSCALE
                self.frame_buffer.append(sub_frame)

            self.img_index = 0
            print("Done -Length:",len(self.action_list))

        if self.img_index >= len(self.file_list)-1:
            gameover_bool = True
            print("Attempting Game-Over")
        current_image = self.frame_buffer[self.img_index]
        xy = int(504/4)
        current_image = np.reshape(current_image, (xy, xy, 1))
        current_human_action = int(self.action_list[self.img_index]) #gotta be int type
        self.img_index += 1
        return current_image, current_human_action, gameover_bool

    #Master reward function. calls reward subfunctions.
    def reward_scan(self, image, mode, agent_action=0):
        """

        """
        #First, let's track what place we're in
        place = 4 #anything past 4th is the same as 4th.

        sixth_features = []
        sixth_features.append(image[113-52-1][276-104-1]) #6th_1
        sixth_features.append(image[182-52-1][221-104-1]) #6th_2
        sixth_features.append(image[230-52-1][248-104-1]) #6th_3
        sixth_truth = 0
        for feature in sixth_features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                sixth_truth += 1
            if sixth_truth == 3:
                place = 6

        first_features = [] #1st Features in Y/X format (pixel fingerprints).
        first_features.append(image[132-52-1][352-104-1]) #1st_1
        first_features.append(image[140-52-1][347-104-1]) #1st_2
        first_features.append(image[146-52-1][309-104-1]) #1st_3
        first_truth = 0
        for feature in first_features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                first_truth += 1
            if first_truth == 3:
                place = 1

        second__features = []
        second__features.append(image[122-52-1][371-104-1]) #2nd_1
        second__features.append(image[150-52-1][355-104-1]) #2nd_2
        second_truth = 0
        for feature in second__features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                second_truth += 1
            if second_truth == 2:
                place = 2

        third__features = []
        third__features.append(image[150-52-1][306-104-1]) #3rd_1
        third__features.append(image[101-52-1][308-104-1]) #3rd_2
        third_truth = 0
        for feature in third__features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                third_truth += 1
            if third_truth == 2:
                place = 3

        wrong_way_crop = image[352-52:422-52, 859-104:1046-104]# Wrong Way Coords in [Y1:Y2, X1:X2] format
        wrong_way_bool = self.wrong_way(wrong_way_crop)

        #Call minmap/speedgauge
        if mode == 0:
            #speedgauge = image[1170-self.y_offset:1400-self.y_offset, 1610-self.x_offset:1860-self.x_offset]
            speedgauge = image[1260-self.y_offset:1360-self.y_offset, 1720-self.x_offset:1820-self.x_offset]
            reward = self.speed_scan(speedgauge)
        elif mode == 1:
            reward = self.minmap_scan(image)
        elif mode == 2:
            reward = self.show_me_scan(agent_action)
        
        return reward, wrong_way_bool, place

    #Scan Minimap to determine reward(on/off track).
    def minmap_scan(self, image):
        """
        We sample pixels in a trident around the front of the minmap green arrow, as well as directly behind.
        - If left and right sides are on track, +1.2, otherwise track yields +0.7
        """
        minmap_reward = -1
        #Front of the green arrow
        minmap_front_image = image[1206-52:1211-52, 1697-104:1702-104]# Unit Label Coords in [Y1:Y2, X1:X2] format
        average_front_img = self.pixel_averager(minmap_front_image)
        #Arrow Left
        minmap_left_image = image[1230-52:1231-52, 1688-104:1689-104]# Unit Label Coords in [Y1:Y2, X1:X2] format
        average_left_img = self.pixel_averager(minmap_left_image)
        #Arrow Right
        minmap_right_image = image[1206-52:1211-52, 1697-104:1702-104]# Unit Label Coords in [Y1:Y2, X1:X2] format
        average_right_img = self.pixel_averager(minmap_right_image)
        #Rear (RED & BLUE ONLY)
        minmap_rear_image = image[1263-52:1265-52, 1699-104:1700-104]# Unit Label Coords in [Y1:Y2, X1:X2] format
        average_rear_img = self.pixel_averager(minmap_rear_image)

        #Filters for track and dots on track.
        track_lower= np.array([0, 0, 80])
        track_upper = np.array([75, 50, 210])
        black_lower= np.array([110, 0, 0])
        black_upper = np.array([200, 250, 35])
        blue_lower= np.array([110, 50, 25])
        blue_upper = np.array([130, 255, 230])
        red_lower= np.array([150, 150, 100])
        red_upper = np.array([190, 255, 220])

        track_mask_front = cv2.inRange(average_front_img, track_lower, track_upper)
        track_mask_left = cv2.inRange(average_left_img, track_lower, track_upper)
        track_mask_right = cv2.inRange(average_right_img, track_lower, track_upper)

        blue_mask_front = cv2.inRange(average_front_img, blue_lower, blue_upper)
        blue_mask_left = cv2.inRange(average_left_img, blue_lower, blue_upper)
        blue_mask_right = cv2.inRange(average_right_img, blue_lower, blue_upper)
        blue_mask_rear = cv2.inRange(average_rear_img, blue_lower, blue_upper)

        red_mask_front = cv2.inRange(average_front_img, red_lower, red_upper)
        red_mask_left = cv2.inRange(average_left_img, red_lower, red_upper)
        red_mask_right = cv2.inRange(average_right_img, red_lower, red_upper)
        red_mask_rear = cv2.inRange(average_rear_img, red_lower, red_upper)

        black_mask_front = cv2.inRange(average_front_img, black_lower, black_upper)
        black_mask_left = cv2.inRange(average_left_img, black_lower, black_upper)
        black_mask_right = cv2.inRange(average_right_img, black_lower, black_upper)
        black_mask_rear = cv2.inRange(average_rear_img, black_lower, black_upper)

        if (track_mask_front[0][0] != 0) or (track_mask_left[0][0] != 0) or (track_mask_right[0][0] != 0):
            minmap_reward = 0.5
            if (track_mask_left[0][0] != 0) and (track_mask_right[0][0] != 0):
                minmap_reward = 1.2
        if (black_mask_front[0][0] != 0) or (black_mask_left[0][0] != 0) or (black_mask_right[0][0] != 0):
            minmap_reward = 1.5
        if (blue_mask_front[0][0] != 0) or (blue_mask_left[0][0] != 0) or (blue_mask_right[0][0] != 0):
            minmap_reward = 1.5
        if (red_mask_front[0][0] != 0) or (red_mask_left[0][0] != 0) or (red_mask_right[0][0] != 0):
            minmap_reward = 1.5
        
        # only give rear points when we're not off track in the front & sides
        if minmap_reward > 0:
            if (black_mask_rear[0][0] != 0) or (blue_mask_rear[0][0] != 0) or (red_mask_rear[0][0] != 0):
                minmap_reward = 1.5
        return minmap_reward

    #OR Scan speedgauge to determine reward(fast/slow).
    def speed_scan(self, img):
        x_mid, y_mid = int(img.shape[1]/2), int(img.shape[0]/2)
        #min_reward, max_reward = -0.3, 1 #-0.75, 4 
        min_reward, max_reward = 0, 5 #-0.75, 4 
        in_min, in_max = -225, -45
        log_min, log_max = 0.001, 0.99

        def cart2pol(x, y):
            """
            Interprets the x,y into degrees around the center.
            Note: less degrees means faster.
            """
            x,y = x-x_mid, (y*-1)+y_mid #note: y values are goofy in array notation.
            #rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            phi = math.degrees(phi) #phi is in raidans, we want degrees.
            if phi < 0:
                phi = 360-abs(phi)
            return phi

        def log_speed(speed):
            speed = round(np.interp(speed*-1,[in_min, in_max],[log_min, log_max]),2) #normalize our inverted polar degree scale to 0-1
            x = (float(speed) / float((log_max)) * 100.0) + 0.99
            base = 10
            log_reward = max(0.0, math.log(x, base) / 2.0)
            return round(log_reward, 4)

        lower = np.array([160,160,160]) #rgb [160,160,160]
        #lower = np.array([240,240,240]) #rgb [160,160,160]
        upper = np.array([250,250,250]) #rgb [250,250,250]

        mask = cv2.inRange(img, lower, upper)
        mask_sum = int(np.sum(mask)/1000)

        matches = np.argwhere(mask==255)
        x, y = matches[:,1], matches[:,0]
        if len(x) > 0 and len(y) > 0: #speedguage isnt blank
            x, y = int(np.mean(x)), int(np.mean(y))
        else:
            x,y = 1,-1 #blank speedguage = lowest reward state.

        phi = cart2pol(x, y)
        if np.isnan(phi):
            phi = 360

        #Log Settings
        phi_log = log_speed(phi)
        speed_reward = round(np.interp(phi_log,[log_min, log_max],[min_reward, max_reward]),2) #map from log-phi to our desired reward range.
        
        #speed_reward = round(np.interp(phi*-1,[in_min, in_max],[min_reward, max_reward]),2) #note: we need to invert phi since angle goes from big to small.
        cv2.imshow("cv2screen", img)
        cv2.waitKey(1)
        if mask_sum > 250:
            speed_reward = min_reward

        return speed_reward

    #Check current keypresses to see if the agent is matching. Need Agent action to score.
    def show_me_scan(self, agent_action):
        #self.action_set = [0'Forward', 1'Powerup', 2'Reverse', 3'Left', 4'Right']
        first_degree = {0:[1,3,4], 1:[0,3,4], 2:[], 3:[0,1], 4:[0,1]} #shows proximity of acceptable alternatives.

        #Find out what key we pressed.
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

        #Agent Guessed right
        if agent_action == master_action:
            self.show_me_stale = 0
            reward = 1

        #Agent wrong. Let's check for partial credit.
        elif agent_action != master_action:
            if agent_action in first_degree[master_action]:
                reward = 0.25
                #Proximity reward decays over 10 instances to -0.25 reward floor.
                self.show_me_stale += 1
                stale_modifier = -0.5*(self.show_me_stale/10)
                reward = reward + max(stale_modifier,-0.5) #cap the floor at -0.25

            else:
                reward = -0.25

        return reward
        
    #determine if a valid item is available.
    def item_scan(self, img):
        item_bool = False
        item_features = []
        item_color = img[1285-52-1][285-104-1] #Item Color
        item_features.append(img[1317-52-1][248-104-1]) #White Circle 1
        item_features.append(img[1360-52-1][202-104-1]) #White Circle 2
        item_features.append(img[1360-52-1][297-104-1]) #White Circle 3
        item_truth = 0

        #Green is avoided
        lower = np.array([0,140,0])
        upper = np.array([10,170,10])
        
        item_color = item_color.astype(np.intc)
        item_color = item_color.reshape(1,1,3)
        mask = cv2.inRange(item_color, lower, upper)

        for feature in item_features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                item_truth += 1
            if item_truth == 3 and mask[0][0]!=255:
                item_bool = True
        return item_bool

    def wrong_way(self, crop):
        wrong_way_bool = True
        features = []
        features.append(crop[32][17]) # Feature 1 in Y/X format.
        features.append(crop[37][54])
        features.append(crop[48][85])
        features.append(crop[25][138])
        features.append(crop[17][166])
        for feature in features:
            if feature[0] < 225 or feature[1] < 225 or feature[2] < 225: #if any of R,G, or B aren't white...
                wrong_way_bool = False
        return wrong_way_bool

    def race_over(self, crop):
        race_over_bool = True
        features = []
        features.append(crop[20][26]) # Feature 1 in Y/X format.
        features.append(crop[120][100])
        features.append(crop[60][150])
        features.append(crop[40][220])
        features.append(crop[125][270]) 
        features.append(crop[25][380])

        for feature in features:
            if feature[0] < 225 or feature[1] < 225 or feature[2] < 225: #if any of R,G, or B aren't white...
                race_over_bool = False
        return race_over_bool

    def side_glitch(self, image):

        side_glitch_bool = False
        side_glitch_image_left = image[948-52:952-52, 888-104:892-104]# Unit Label Coords in [Y1:Y2, X1:X2] format
        side_glitch_average_img_left = self.pixel_averager(side_glitch_image_left)
        side_glitch_image_right = image[948-52:952-52, 1138-104:1142-104]# Unit Label Coords in [Y1:Y2, X1:X2] format
        side_glitch_average_img_right = self.pixel_averager(side_glitch_image_right)

        side_glitch_lower= np.array([10, 122, 28])
        side_glitch_upper = np.array([20, 135, 36])
        obelisk_lower= np.array([18, 125, 100])
        obelisk_upper = np.array([24, 135, 205])
        tent_lower= np.array([0, 0, 0])
        tent_upper = np.array([2, 2, 2])

        side_glitch_mask_left = cv2.inRange(side_glitch_average_img_left, side_glitch_lower, side_glitch_upper)
        side_glitch_mask_right = cv2.inRange(side_glitch_average_img_right, side_glitch_lower, side_glitch_upper)
        obelisk_mask_left = cv2.inRange(side_glitch_average_img_left, obelisk_lower, obelisk_upper)
        obelisk_mask_right = cv2.inRange(side_glitch_average_img_right, obelisk_lower, obelisk_upper)
        tent_mask_left = cv2.inRange(side_glitch_average_img_left, tent_lower, tent_upper)
        tent_mask_right = cv2.inRange(side_glitch_average_img_right, tent_lower, tent_upper)

        if (side_glitch_mask_left[0][0] != 0) or (obelisk_mask_left[0][0] != 0) or (tent_mask_left[0][0] != 0):
            side_glitch_bool = True
        if (side_glitch_mask_right[0][0] != 0) or (obelisk_mask_right[0][0] != 0) or (tent_mask_right[0][0] != 0):
            side_glitch_bool = True

        return side_glitch_bool

    def pixel_averager(self, image_crop):
        #input the cropped image
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV) # Convert to HSV
        average_pixel = np.mean(image_crop, axis = 0)
        average_pixel = np.mean(average_pixel, axis = 0)
        average_pixel = average_pixel.astype(np.intc)
        average_img = average_pixel.reshape(1,1,3)

        return average_img

# Where neural outputs are covnerted to actions
class Actions():

    def __init__(self):
        self.last_action = int(5) #initiate out of range to be overwritten.
        #modes: 1:Circuit 2:Single, 3:Versus, 4:Time
        #course 2 track 3 is the problem level.
        self.mode, self.course, self.track = 4, 2, 3 #mode/course/track selection.

    # Map NN output to actions
    def action_Map(self, action):

        key_mapper.actionMap(action, self.last_action)
        
        self.last_action = action

    # Reset game state
    def Reset(self):
        key_mapper.reset()
        time.sleep(5.5)
        print('GO!')
        return

    def quit_Reset(self):
        key_mapper.quitReset()

    def first_Run(self):
        key_mapper.firstRun()

    def toggle_pause(self, bool):
        if bool:
            pydirectinput.press('enter') # Simulate pressing the Enter key.
        if not bool:
            pydirectinput.press('esc') # Simulate pressing the Escape key.

# Initialization Parameters for Troubleshooting.
def main(mode):

    if mode == 'Actions':
        act = Actions()
        act.Reset()
        i = 0
        while i < 100:
            print(f'Iteration: {i}')
            action = np.random.randint(0, 5)
            act.action_Map(action)
            i+=1
            time.sleep(0.2)

    elif mode == 'Minmap':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            print(img.shape)
            grab.minmap_Scan(img)

    elif mode == 'Speed':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            speedgauge = img[1260-52-1:1360-52-1, 1720-104-1:1820-104-1]
            reward = grab.speed_scan(speedgauge)
            print(reward)

    elif mode == 'Reset':
        time.sleep(1)
        act = Actions()
        act.Reset()

    elif mode == 'Place':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            _, _, place = grab.reward_scan(img, 0)
            print(place)

    elif mode == 'Item':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            item_bool = grab.item_scan(img)
            print(item_bool)

    elif mode == 'Finish':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            img = img[300-52:440-52,810-104:1220-104]
            grab.race_over(img)

    elif mode == 'First':
        time.sleep(1)
        act = Actions()
        act.first_Run()

    elif mode == 'Quit':
        time.sleep(1)
        act = Actions(0)
        i = 0
        while i < 10:
            act.quit_Reset()
            i+=1

if __name__ == '__main__':
    main('First')