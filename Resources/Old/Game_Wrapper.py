import time
import random
import numpy as np
import cv2
import os
from PIL import ImageGrab, Image, ImageOps
import win32gui, win32ui, win32con, win32api # for quick_Grab

import pyautogui
import pydirectinput #this module requires pyautogui to run


import matplotlib.pyplot as plot

'''
Notes:
The gamescreen offset is (52,104).

'''

# Where screencaptures are executed
class ScreenGrab():

    def __init__(self):
        #self.wrong_way_coords = {1:(17,32), 2:(54,37), 3:(85,48), 4:(138,25), 5:(166,17)}
        print('Initializing Screen_Grab')
        self.x_offset = 104-1
        self.y_offset = 52-1

    #Using the inferior method in hopes it preserves the mouse.
    def quick_Grab(self, region=None):
        hwin = win32gui.GetDesktopWindow()
        if region:
            '''
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
            '''
            left, top, width, height = 104, 52, (1922-104 +1), (1472-52  +1)

        else:
            '''
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
            '''
            left, top, width, height = 104, 52, (1922-104 +1), (1472-52  +1)

            
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height,width,4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) #retruns the RGB Image
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #retruns the Grayscale
        return img

    # Label Image to Text
    def minmap_Scan(self, image):
        """
        We sample pixels in a trident around the front of the minmap green arrow, as well as directly behind.
        - If left and right sides are on track, +1.2, otherwise track yields +0.7

        The 'wrong way' warning area is also sampled, and 5 coord points checked to make sure the condition isn't triggered.
        """
        #First, let's track what place we're in
        place = 4 #anything past 4th is the same for us.

        first_features = [] #1st Features in Y/X format (like pixel fingerprints).
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

        #Track check does not included rear.
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

        if wrong_way_bool: #Wrong way overrides any reward.
            minmap_reward = -1.5

        #print(f'Reward: {minmap_reward}        {image[2][2]}   {average_front_pixel}')
        return minmap_reward, wrong_way_bool, place

    # Screenshot functions do not capture mouse; it must be added. OBSOLETE
    def add_Mouse(self, image):
        # reshape and prepare for tensor conversion
        image = np.array(image)
        image = cv2.resize(image, (504, 504))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.reshape(image, (504, 504, 1))

        return image
        
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
        #side_glitch_image = cv2.cvtColor(side_glitch_image, cv2.COLOR_RGB2HSV) # Convert to HSV
        #print(f'{side_glitch_image[0][0]}')

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
        #print(f'{side_glitch_bool}')
        return side_glitch_bool

    def pixel_averager(self, image_crop):
        #input the cropped image
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV) # Convert to HSV
        average_pixel = np.mean(image_crop, axis = 0)
        average_pixel = np.mean(average_pixel, axis = 0)
        average_pixel = average_pixel.astype(np.intc)
        average_img = np.concatenate((average_pixel, average_pixel, average_pixel, average_pixel), axis=0)
        average_img = average_img.reshape(2,2,3)
        return average_img

# Where neural outputs are covnerted to actions
class Actions():

    def __init__(self):
        print('Initializing Actions')
        self.last_action = 5 #initiate out of range to be overwritten.
        pydirectinput.keyDown('w')
        pyautogui.PAUSE = 0.001 #this controls time betweena ctions.

    # Map NN output to actions
    def action_Map(self, action):
        '''
        A multitrhread process will be spawned every time an action is received. If the action is the 
        same as the last action, do nothing and let it ride. 
        All actions include moving forward by default, except for the reverse action which interrupts forward.
        '''
        action_label = {0:'Do Nothing', 1:'Fire Powerup', 2:'Reverse', 3:'Left', 4:'Right'}
        action_key = {0:'0', 1:'q', 2:'s', 3:'a', 4:'d', 5: 'NULL'}
        #print(action_label[action])

        if self.last_action == action: # we don't need to do anything for repeats.
            return

        elif self.last_action != action:
            pydirectinput.keyUp(action_key[self.last_action])
            # terminate old multithread
            # launch new multithread
            self.last_action = action
            pyautogui.keyDown('w')

            if action == 0: # Do Nothing
                pass
            elif action == 1: # fire power up
                pydirectinput.press('q')

            elif action == 2: # DOWN
                pydirectinput.keyUp('w')
                pydirectinput.keyDown('s')

            elif action == 3: # LEFT
                pydirectinput.keyDown('a')

            elif action == 4: # RIGHT
                pydirectinput.keyDown('d')


    # Reset game state;
    def Reset(self):
        pydirectinput.keyUp('w')
        pyautogui.press('esc')
        time.sleep(0.03)
        pydirectinput.press('down')
        time.sleep(0.03)
        pyautogui.press('enter')
        time.sleep(0.03)
        pydirectinput.press('up')
        time.sleep(0.03)
        pyautogui.press('enter')
        pyautogui.keyDown('w')
        time.sleep(5)
        print('GO!')

# Initialization Parameters for Troubleshooting.
def main(mode):

    if mode == 'Actions':
        act = Actions()
        i = 0
        start = time.time()
        while i < 1000:
            action = np.random.randint(0, 5)
            act.action_Map(action)
            i+=1
        end = time.time()
        total_time = end-start
        adj_time = time.strftime("%S", time.gmtime(total_time))
        print(adj_time)

    elif mode == 'Minmap':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            grab.minmap_Scan(img)

    elif mode == 'Mouse':
        pass
    
    elif mode == 'Reset':
        act = Actions()
        act.Reset()

    elif mode == 'Corner':
        grab = ScreenGrab()

        while 1:
            img = grab.quick_Grab()
            grab.side_glitch(img)

    elif mode == 'Finish':
        grab = ScreenGrab()
        while 1:
            img = grab.quick_Grab()
            img = img[300-52:440-52,810-104:1220-104]
            grab.race_over(img)

if __name__ == '__main__':
    main('Actions')