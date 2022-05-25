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

# Where screencaptures are executed (20fps speed)
#800x600 old
class ScreenGrab():

    def __init__(self, dataset_mode = "Solo"):
        self.xy = 126  #downsize to 126x126px
        self.x_offset = 104-1
        self.y_offset = 52-1
        self.show_me_stale = 0
        self.dataset_mode = dataset_mode #Solo, Group
        self.home_dir = os.getcwd()

        watermark_up = cv2.imread(r"Resources\1_up.png", -1) #-1 is neccessary for alpha.
        watermark_down = cv2.imread(r"Resources\1_down.png", -1)
        watermark_left = cv2.imread(r"Resources\1_left.png", -1)
        watermark_right = cv2.imread(r"Resources\1_right.png", -1)
        watermark_circle = cv2.imread(r"Resources\1_circle.png", -1)
        self.watermark_dict = {0:watermark_up, 1:watermark_circle, 2:watermark_down, 
                                3:watermark_left, 4:watermark_right}

    #Quick Screenshot
    def quick_Grab(self, region=None):
        hwin = win32gui.GetDesktopWindow()
        if region:
            #left, top, width, height = 104, 52, (1922-104 +1), (1472-52  +1)
            left, top, width, height = 104, 52, 1819, 1421

        else:
            #left, top, width, height = 104, 52, (1922-104 +1), (1472-52  +1)
            #left, top, width, height = 104, 52, 1819, 1421 #1819x1421px
            #left, top, width, height = 846, 389, 305, 352 #305x352px
            left, top, width, height = 45, 344, 873, 925 #873x925px

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
        img = np.array(img)
        img = cv2.resize(img, (self.xy, self.xy))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img) #equalize the image for greater contrast.
        if wrong_way_bool: #invert image if we're going the wrong way.
            img = ~img
        img = np.reshape(img, (self.xy, self.xy, 1))
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
            if self.dataset_mode == "Solo":
                os.chdir("Dataset\Solo")
            elif self.dataset_mode == "Group":
                os.chdir("Dataset\Group")
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
                sub_frame = cv2.cvtColor(sub_frame, cv2.COLOR_GRAY2RGB) #Alhpa channel for watermark
                #sub_frame = np.reshape(sub_frame, (self.xy, self.xy, 1))
                self.frame_buffer.append(sub_frame)

            self.img_index = 0
            print("Done -Length:",len(self.action_list))

        if self.img_index >= len(self.file_list)-1:
            gameover_bool = True
            print("Attempting Game-Over")
        current_image = self.frame_buffer[self.img_index]
        current_human_action = int(self.action_list[self.img_index]) #gotta be int type
        self.img_index += 1
        return current_image, current_human_action, gameover_bool

    #Master reward function. calls reward subfunctions.
    def reward_scan(self, image, mode, agent_action=0):

        """let's track what place we're in
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
        """
        place = 1 #this is a dummy value

        wrong_way_crop = image[195:241, 362:513]# Wrong Way Coords in [Y1:Y2, X1:X2] format
        #status = cv2.imwrite("wrong_way_sample.png", wrong_way_crop)
        wrong_way_bool = self.wrong_way(wrong_way_crop)

        #Call Reward Mode
        if mode == 0: #Minmap
            #minmap_img = image[651:893, 680:850] #the tuning for the classic mode.
            adj1, adj2 = 90,65
            minmap_img = image[741:803, 745:785]
            reward = self.minmap_scan(minmap_img)
        elif mode == 1:#Speedguage
            #speedgauge = image[1260-self.y_offset:1360-self.y_offset, 1720-self.x_offset:1820-self.x_offset]
            speedgauge = image[786:851, 775:823]#873x925px
            reward = self.speed_scan(speedgauge)
        elif mode == 2:#ShowMeLive
            reward = self.show_me_scan(agent_action)
        elif mode == 3:#dummy mode
            reward=0
        return reward, wrong_way_bool, place

    # Scan the minimap for reward (on/off track; rivals proximity)
    def minmap_scan(self, image):
        """
        we use a similar scheme to the speedcheck protocol. Most checks should be infront of the green race arrow,
        so the image for those is the top 50 of img. 
        """
        #Front of the green arrow

        x_mid, y_mid = int(image.shape[1]/2), int(image.shape[0]/2)
        total_area = image.shape[1]*image.shape[0]
        little_y = y_mid-20 #cropping out part of the racecar arrow.
        image_track_top = image[:little_y,:]
        image_top = image[:y_mid,:]
        min_reward = -1

        #Filters for track and dots on track.
        adj = 25
        track_lower= np.array([192-adj, 192-adj, 192-adj]) #(192,192,192)
        track_upper = np.array([192+35, 192+35, 192+adj]) #(192,192,192)
        blue_lower= np.array([40-adj, 0, 216-adj]) #(40,0,216)
        blue_upper = np.array([40+adj, 0+adj, 216+adj]) #(40,0,216)
        red_lower= np.array([216-adj,0,40-adj]) #(216,0,40)
        red_upper = np.array([216+adj,0,40+adj]) #(216,0,40)


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

        track_mask = cv2.inRange(image_track_top, track_lower, track_upper)        
        blue_front_mask = cv2.inRange(image_top, blue_lower, blue_upper)
        red_front_mask = cv2.inRange(image_top, red_lower, red_upper)
        
        track_matches = np.argwhere(track_mask==255)
        blue_front_matches = np.argwhere(blue_front_mask==255)
        red_front_matches = np.argwhere(red_front_mask==255)

        x, y = track_matches[:,1], track_matches[:,0]
        if len(x) > 0 and len(y) > 0: #speedguage isnt blank
            x, y = int(np.mean(x)), int(np.mean(y))
        else:
            x,y = 1,-1 #blank speedguage = lowest reward state.

        phi = cart2pol(x, y)
        if np.isnan(phi):
            phi = 360

        """The ideal range we want is 90deg, meaning the track is right infront of us (think _| )"""
        #minmap_reward = round(np.interp(phi*-1,[in_min, in_max],[min_reward, max_reward]),2) #note: we need to invert phi since angle goes from big to small.
        minmap_reward = -1
        target = 90
        variance_1, variance_2 = 10, 25 #degree variance +/- from the target
        
        #first let's see if we have a boost from the rival or other racers:
        if len(red_front_matches)/(total_area/2) > 0.03: #blue dots take up > 5% of screen
            minmap_reward = 1.5
            #print("redmatch", end="")

        elif len(blue_front_matches)/(total_area/2) > 0.03: #blue dots take up > 5% of screen
            minmap_reward = 1.5
            #print("bluematch", end="")
        
        #No other racers, let's look for track
        elif minmap_reward < 0:
            if (phi > target-variance_1) and (phi < target+variance_1):
                minmap_reward = 1.2
            elif (phi > target-variance_2) and (phi < target+variance_2):
                minmap_reward = 0.5

            if minmap_reward > 0: #check for racers behind agent
                blue_rear_mask = cv2.inRange(image_top, blue_lower, blue_upper)
                red_rear_mask = cv2.inRange(image_top, red_lower, red_upper)
                blue_rear_matches = np.argwhere(blue_rear_mask==255)
                red_rear_matches = np.argwhere(red_rear_mask==255)
                #first let's see if we have a boost from the rival or other racers:
                if len(red_rear_matches)/(total_area/2) > 0.03: #blue dots take up > 5% of screen
                    minmap_reward = 1.5
                    #print("redmatch", end="")
                elif len(blue_rear_matches)/(total_area/2) > 0.03: #blue dots take up > 5% of screen
                    minmap_reward = 1.5

        #cv2.imshow("cv2screen", red_mask)
        #cv2.waitKey(1)

        if len(track_matches)/(total_area/2) > 0.75: #Something is wrong if 75% of mask is track.
            minmap_reward = min_reward

        #print(minmap_reward, phi)
        return minmap_reward

    # Scan speedgauge for reward(fast/slow).
    def speed_scan(self, img):
        x_mid, y_mid = int(img.shape[1]/2), int(img.shape[0]/2)
        min_reward, max_reward = -3, 2 #the floor should be below the min for log adjustment
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
        #cv2.imshow("cv2screen", img)
        #cv2.waitKey(1)
        if mask_sum > 250:
            speed_reward = min_reward

        return speed_reward

    def item_scan(self, img): #determine if a valid item is available.
        #feed full screenshot. Detects all but level 4 yellow mummies curse.
        item_bool = False
        item_features = []
        item_color = img[802][86] #Item Color
        item_features.append(img[823][69]) #White Circle Top Middle
        item_features.append(img[865][48]) #White Circle Lower Left
        #item_features.append(img[851][47]) #White Circle 2
        #item_features.append(img[882][69]) #White Circle 2
        item_features.append(img[851][92]) #White Circle Lower Right
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
            #if item_truth == 3 and mask[0][0]!=255: #use this to avoid speed boosts, which are hard to control.
            #    item_bool = True
            if item_truth == 3:
                item_bool = True       
        return item_bool

    def place_scan(self, img):
        #img crop = [87:88, 51:87]
        #take a 1x36 crop and if the sum of the match indexies == the fingerprint we have a total match
        place = 6
        adj = 15
        match_sum = [0,0]
        
        white_lower= np.array([255-adj, 255-adj, 255-adj])
        white_upper = np.array([256, 256, 256]) 
        white_mask = cv2.inRange(img, white_lower, white_upper)
        white_matches = np.argwhere(white_mask==255)

        if len(white_matches) > 0: #we can only sum non-empty arrays.
            match_sum = list(sum(white_matches))
        fingerprints = {1:[0, 153], 2:[0, 69], 3:[0, 329], 4:[0, 490],5:[0, 212],6:[0,147]}

        for place_key, signature in fingerprints.items():
            if match_sum == signature:
                place = place_key

        #print(place)
        return place

    def lap_scan(self, img):
        lap = 0 #dummy value to catch errors

        first_lap_features = []
        first_lap_features.append(img[20][7])# Feature 1 in Y/X format.
        first_lap_features.append(img[15][27])
        first_lap_features.append(img[12][45])
        first_lap_truth = 0
        for feature in first_lap_features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                first_lap_truth += 1
            if first_lap_truth == 3:
                lap = 1

        second_lap_features = []
        second_lap_features.append(img[42][1])
        second_lap_features.append(img[11][55])
        second_lap_features.append(img[42][63])
        second_lap_truth = 0
        for feature in second_lap_features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                second_lap_truth += 1
            if second_lap_truth == 3:
                lap = 2

        third_lap_features = []
        third_lap_features.append(img[11][1])
        third_lap_features.append(img[14][23])
        third_lap_features.append(img[36][66])
        third_lap_truth = 0
        for feature in third_lap_features:
            if feature[0] > 225 and feature[1] > 225 and feature[2] >225: #if R,G, and B are white...
                third_lap_truth += 1
            if third_lap_truth == 3:
                lap = 3

        return lap

    def go_scan(self, img):
        #We're looking for yellow (248,252,0)
        go_bool = False
        first_lap_features = []
        first_lap_features.append([img[77][13]])# Feature 1 in Y/X format.
        first_lap_features.append([img[13][42]])
        first_lap_features.append([img[150][42]])
        first_lap_features.append([img[13][104]])
        first_lap_features.append([img[150][104]])
        first_lap_features.append([img[77][133]])
        first_lap_features = np.array(first_lap_features)
        #print(img.shape, first_lap_features.shape)

        adj = 25
        yellow_lower= np.array([248-adj,252-adj,0]) #(216,0,40)
        yellow_upper = np.array([248+adj,252+adj,0]) #(216,0,40)

        yellow_mask = cv2.inRange(first_lap_features, yellow_lower, yellow_upper)
        yellow_matches = np.argwhere(yellow_mask==255)
        if len(yellow_matches) == len(first_lap_features): #they all matched
            go_bool = True
        
        return go_bool

    def wrong_way(self, crop):
        wrong_way_bool = True
        features = []
        features.append(crop[11][7]) # Feature 1 in Y/X format.
        features.append(crop[23][26])
        features.append(crop[30][40])
        features.append(crop[22][58])
        features.append(crop[12][80])
        for feature in features:
            if feature[0] < 225 or feature[1] < 225 or feature[2] < 225: #if any of R,G, or B aren't white...
                wrong_way_bool = False
        return wrong_way_bool

    def race_over(self, crop):
        race_over_bool = True
        features = []
        features.append(crop[16][20]) # Feature 1 in Y/X format.
        features.append(crop[16][46])
        features.append(crop[40][84])
        features.append(crop[40][125])
        features.append(crop[40][160]) 
        features.append(crop[40][105])

        for feature in features:
            if feature[0] < 225 or feature[1] < 225 or feature[2] < 225: #if any of R,G, or B aren't white...
                race_over_bool = False
        return race_over_bool

    def side_glitch(self, image):
        side_glitch_bool = False
        side_glitch_image_left = image[583:586, 376:378]# Unit Label Coords in [Y1:Y2, X1:X2] format
        side_glitch_average_img_left = self.pixel_averager(side_glitch_image_left)
        side_glitch_image_right = image[583:586, 496:498]# Unit Label Coords in [Y1:Y2, X1:X2] format
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

        #tree glitch section
        tree_glitch_image_left = image[473:501, 426:437]# Unit Label Coords in [Y1:Y2, X1:X2] format
        tree_glitch_average_img = self.pixel_averager(tree_glitch_image_left)
        
        adj = 25
        tree_lower= np.array([0,226-adj,71-adj]) #rgb (54,28,8) (15,226,71)
        tree_upper= np.array([15+adj,226+adj,71+adj]) #rgb (54,28,8)
        tree_mask_left = cv2.inRange(tree_glitch_average_img, tree_lower, tree_upper)
        if (tree_mask_left[0][0] != 0):
            side_glitch_bool = True
            #print(tree_glitch_average_img[0][0], end="")

        return side_glitch_bool

    def pixel_averager(self, image_crop):
        #input the cropped image
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV) # Convert to HSV
        average_pixel = np.mean(image_crop, axis = 0)
        average_pixel = np.mean(average_pixel, axis = 0)
        average_pixel = average_pixel.astype(np.intc)
        average_img = average_pixel.reshape(1,1,3)

        return average_img

    def add_watermark(self, image, action):
        #Note: Watermak img must be loaded as cv2.imread("watermark.png", -1) #-1 is neccessary.
        watermark = self.watermark_dict[action]
        h_img, w_img, _ = image.shape
        h_wm, w_wm, _ = watermark.shape

        y = 0
        x = int(w_img/2)-int(w_wm/2) #center of img offset by the center of the watermark.

        image_overlay = watermark[:, :, 0:3]
        alpha_mask = watermark[:, :, 3] / 255.0

        # Image ranges
        y1, y2 = max(0, y), min(image.shape[0], y + image_overlay.shape[0])
        x1, x2 = max(0, x), min(image.shape[1], x + image_overlay.shape[1])
        # Overlay ranges
        y1o, y2o = max(0, -y), min(image_overlay.shape[0], image.shape[0] - y)
        x1o, x2o = max(0, -x), min(image_overlay.shape[1], image.shape[1] - x)
        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return print('ScreenGrab.add_Mouse FAILED')
        channels = image.shape[2]
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha
        for c in range(channels):
            image[y1:y2, x1:x2, c] = (alpha * image_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * image[y1:y2, x1:x2, c])

        return image

    def add_watermark_dataset(self, image, action):
        #add watermark to small images from database.
        #Note: Watermak img must be loaded as cv2.imread("watermark.png", -1) #-1 is neccessary.
        watermark = self.watermark_dict[action]
        h_img, w_img, _ = image.shape
        h_wm, w_wm, _ = watermark.shape

        #Resize watermark from 873x925px to 126x126 scale
        new_x = round(np.interp(w_wm,[0, 873],[0, 126])) #map from log-phi to our desired reward range.
        new_y = round(np.interp(h_wm,[0, 925],[0, 126])) #map from log-phi to our desired reward range.
        watermark = cv2.resize(watermark, (new_x, new_y)) #(width, height)
        h_wm, w_wm, _ = watermark.shape

        y = 0
        x = int(w_img/2)-int(w_wm/2) #center of img offset by the center of the watermark.

        image_overlay = watermark[:, :, 0:3]
        alpha_mask = watermark[:, :, 3] / 255.0

        # Image ranges
        y1, y2 = max(0, y), min(image.shape[0], y + image_overlay.shape[0])
        x1, x2 = max(0, x), min(image.shape[1], x + image_overlay.shape[1])
        # Overlay ranges
        y1o, y2o = max(0, -y), min(image_overlay.shape[0], image.shape[0] - y)
        x1o, x2o = max(0, -x), min(image_overlay.shape[1], image.shape[1] - x)
        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return print('ScreenGrab.add_Mouse FAILED')
        channels = image.shape[2]
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha
        for c in range(channels):
            image[y1:y2, x1:x2, c] = (alpha * image_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * image[y1:y2, x1:x2, c])

        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY) #grayscale
        image = np.reshape(image, (self.xy, self.xy, 1))
        return image

# Where neural outputs are covnerted to actions
class Actions():

    def __init__(self):
        self.last_action = int(5) #initiate out of range to be overwritten.
        #modes: 1:Circuit 2:Single, 3:Versus, 4:Time
        #course 2 track 3 is the problem level.
        self.mode, self.course, self.track = 4, 2, 3 #mode/course/track selection.
        self.current_mapmode = 0 #minmap(0), speedguage(1)

    # Map NN output to actions
    def action_Map(self, action):
        key_mapper.actionMap(action, self.last_action)
        self.last_action = action

    # Reset game state
    def Reset(self):
        grab = ScreenGrab()
        key_mapper.reset()
        while 1: #Check to see when its time to go.
            img = grab.quick_Grab()
            go_crop = img[154:316, 363:511]# Wrong Way Coords in [Y1:Y2, X1:X2] format
            go_bool = grab.go_scan(go_crop)
            if go_bool:
                print('GO!')
                return

    def quit_Reset(self):
        key_mapper.quitReset()

    def first_Run(self):
        key_mapper.firstRun()

    def toggle_pause(self, bool):
        key_mapper.togglePause(bool)

    def toggle_mode(self):
        if self.current_mapmode: #Currently Speedguage
            #print("Speedguage2Minmap")
            key_mapper.toggleMode(self.current_mapmode)
            self.current_mapmode = 0 #set to minmap
            
        elif not self.current_mapmode: #Currently Minmap
            #print("Minmap2Speedguage")
            key_mapper.toggleMode(self.current_mapmode)
            self.current_mapmode = 1 #set to speedguage

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

    elif mode == 'Reward':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            reward, wrong_way_bool, place = grab.reward_scan(img,1)
            print(reward, wrong_way_bool, place)

    elif mode == 'Screengrab':
        grab = ScreenGrab()
        print("Starting ScreenGrab")
        #@function_timer
        def screengrab_subfunction():
            for i in range(1000):
                img = grab.quick_Grab()
                #sub_img = img[154:316, 363:511]# Wrong Way Coords in [Y1:Y2, X1:X2] format
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA) #retruns the BGR Image
                status = cv2.imwrite(f"mini_img_sample{i}.png", img)
                #status = cv2.imwrite(f"sub_img{i}.png", sub_img)
                time.sleep(0.1)

        screengrab_subfunction()

    elif mode == 'Minmap':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            adj1, adj2 = 90,65
            minmap_img = img[651+adj1:893-adj1, 680+adj2:850-adj2]
            reward = grab.minmap_scan(minmap_img)

    elif mode == 'Speed':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            reward, wrong_way_bool, place = grab.reward_scan(img,1)
            print(reward, wrong_way_bool, place)

    elif mode == 'toggleMode':
        act = Actions()
        for i in range(100):
            act.toggle_mode()
            #time.sleep(2)
        return

    elif mode == 'Reset':
        time.sleep(1)
        act = Actions()
        act.Reset()

    elif mode == 'Place':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            place_crop = img[87:88, 51:87]
            place = grab.place_scan(place_crop)

    elif mode == 'Lap':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            lap_crop = img[21:72, 632:706]
            lap = grab.lap_scan(lap_crop)
            print(lap)

    elif mode == 'sideGlitch':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            side_glitch_bool = grab.side_glitch(img)
            #print(side_glitch_bool)

    elif mode == 'Item':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            #status = cv2.imwrite(f"mini_img_sample.png",cv2.cvtColor(img, cv2.COLOR_RGB2BGRA))
            item_bool = grab.item_scan(img)
            print(item_bool)

    elif mode == 'Go':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            img = img[154:316, 363:511]# Wrong Way Coords in [Y1:Y2, X1:X2] format
            go_bool = grab.go_scan(img)
            print(go_bool)

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

    elif mode == 'Watermark':
        grab = ScreenGrab()
        print("Starting ScreenGrab")
        #@function_timer
        def screengrab_subfunction():
            for i in range(1000):
                img = grab.quick_Grab()
                img = grab.resize(img)
                action = np.random.randint(5)
                output_img = grab.add_watermark_dataset(img, action)
                cv2.imshow("Watermarked Image", output_img)
                cv2.waitKey(1)

        screengrab_subfunction()

if __name__ == '__main__':
    main('Item')