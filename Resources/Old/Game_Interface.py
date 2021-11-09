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
The gamescreen offset is (113,113).

'''

# Where screencaptures are executed
class ScreenGrab():

    def __init__(self):
        print('Initializing Screen_Grab')

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
        we make a mini crop just above the arrow, and take the target value through different filters.
        black mask accounts for the circle outlines.
        """
        minmap_reward = -1
        image = image[1206-52:1211-52, 1697-104:1702-104]# Unit Label Coords in [Y1:Y2, X1:X2] format
        image= cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # Convert to HSV
        #print(image)
        average_pixel = np.mean(image, axis = 0)
        average_pixel = np.mean(average_pixel, axis = 0)
        average_pixel = average_pixel.astype(np.intc)
        average_img = np.concatenate((average_pixel, average_pixel, average_pixel, average_pixel), axis=0)
        average_img = average_img.reshape(2,2,3)
        #print(average_pixel)

        track_lower= np.array([0, 0, 80])
        track_upper = np.array([75, 50, 210])

        black_lower= np.array([110, 0, 0])
        black_upper = np.array([200, 250, 35])

        blue_lower= np.array([110, 50, 25])
        blue_upper = np.array([130, 255, 230])

        red_lower= np.array([150, 150, 100])
        red_upper = np.array([190, 255, 220])

        track_mask = cv2.inRange(average_img, track_lower, track_upper)
        blue_mask = cv2.inRange(average_img, blue_lower, blue_upper)
        red_mask = cv2.inRange(average_img, red_lower, red_upper)
        black_mask = cv2.inRange(average_img, black_lower, black_upper)

        if track_mask[0][0] != 0:
            minmap_reward = 1

        if black_mask[0][0] != 0:
            minmap_reward = 0.5

        if blue_mask[0][0] != 0:
            minmap_reward = 1.5
            
        if red_mask[0][0] != 0:
            minmap_reward = 2

        #print(f'Reward: {minmap_reward}        {image[2][2]}   {average_pixel}')
        return minmap_reward

    # Screenshot functions do not capture mouse; it must be added. OBSOLETE
    def add_Mouse(self, image):
        # reshape and prepare for tensor conversion
        image = np.array(image)
        image = cv2.resize(image, (504, 504))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.reshape(image, (504, 504, 1))

        return image
        
# Where neural outputs are covnerted to actions
class Actions():

    def __init__(self):
        print('Initializing Actions')
        pydirectinput.keyDown('w')
        pyautogui.PAUSE = 0.001 #this controls time betweena ctions.

    # Map NN output to actions
    def action_Map(self, action):
        '''
        Action [0:Do Nothing, 1:Right Click, 2:Left Click, 3:Change X, 4:Change Y]
        '''
        key_time = 0.2
        pyautogui.keyDown('w')

        if action == 0: # Do Nothing
            time.sleep(key_time)
            print('Do Nothing')
            pass

        elif action == 1: # fire power up
            pydirectinput.keyDown('q')
            time.sleep(key_time)
            pydirectinput.keyUp('q')
            print('Fire Powerup')

        elif action == 2: # DOWN
            pydirectinput.keyUp('w')
            pydirectinput.keyDown('s')
            time.sleep(key_time)
            pydirectinput.keyUp('s')
            pydirectinput.keyDown('w')
            print('Down')

        elif action == 3: # LEFT
            pydirectinput.keyDown('a')
            time.sleep(key_time)
            pydirectinput.keyUp('a')
            print('Left')

        elif action == 4: # RIGHT
            pydirectinput.keyDown('d')
            time.sleep(key_time)
            pydirectinput.keyUp('d')
            print('Right')

        '''
        if action == 0: # Do Nothing
            pydirectinput.keyDown('w')
            time.sleep(key_time)
            pydirectinput.keyUp('w')

            print('Do Nothing')
            pass

        elif action == 1: # fire power up
            pydirectinput.keyDown('w')
            pydirectinput.keyDown('q')
            time.sleep(key_time)
            pydirectinput.keyUp('q')
            pydirectinput.keyUp('w')
            print('Fire Powerup')

        elif action == 2: # DOWN
            pydirectinput.keyDown('s')
            time.sleep(key_time)
            pydirectinput.keyUp('s')
            print('Down')

        elif action == 3: # LEFT
            pydirectinput.keyDown('w')
            pydirectinput.keyDown('a')
            time.sleep(key_time)
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('w')
            print('Left')

        elif action == 4: # RIGHT
            pydirectinput.keyDown('w')
            pydirectinput.keyDown('d')
            time.sleep(key_time)
            pydirectinput.keyUp('d')
            pydirectinput.keyUp('w')
            print('Right')
            '''

    # Reset game state; initialize pointer on box edge.
    def Reset(self):
        pyautogui.keyUp('w')
        pyautogui.press('esc')
        time.sleep(0.03)
        pydirectinput.press('down')
        time.sleep(0.03)
        pyautogui.press('enter')
        time.sleep(0.03)
        pydirectinput.press('up')
        time.sleep(0.03)
        pyautogui.press('enter')
        time.sleep(5)
        print('GO!')

# Initialization Parameters for Troubleshooting.
def main(mode):

    if mode == 'Actions':
        act = Actions()
        i = 0
        while i < 150:
            action = np.random.randint(0, 5)
            act.action_Map(action)


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

if __name__ == '__main__':
    main('Minmap')