import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import cv2
import numpy as np
import math
import time


def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
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
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def pathing(img):
    x_mid, y_mid = int(img.shape[1]/2), int(img.shape[0]/2)
    min_reward, max_reward = -0.3, 1 #-0.75, 4 
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

    phi_log = log_speed(phi)
    speed_reward = round(np.interp(phi_log,[log_min, log_max],[min_reward, max_reward]),2) #map from log-phi to our desired reward range.

    print(f"Sum:{mask_sum}  Reward:{round(speed_reward,2)}")

    cv2.imshow("cv2screen", mask)
    #cv2.imshow("cv2screen", minimap)
    cv2.waitKey(10)

def resize(img):
    size_factor = 4
    dim = 504
    xy = int(dim/size_factor)
    img = np.array(img)
    img = cv2.resize(img, (xy, xy))
    #img = cv2.resize(img, (dim, dim))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img) #equalize the image for greater contrast.
    img = np.reshape(img, (xy, xy, 1))
    #img = np.reshape(img, (dim, dim, 1))

    return img

"""
#countdown
for i in range(3):
    print(i)
    time.sleep(1)
"""


def main(mode):
    if mode == "speed":
        # run for just 100 frames.
        while 1:
            #screen = grab_screen(region=(1280, 0, 3840, 1440))  # region will vary depending on game resolution and monitor resolution
            screen = grab_screen(region=(0, 0, 3840, 2160))  # region will vary depending on game resolution and monitor resolution
            
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB) # because default will be BGR

            base_x, base_y, side_length = 1770, 1310, 50
            #minimap = screen[1200:1400, 1625:1895]
            #minimap = screen[1210:1410, 1665:1875]
            minimap = screen[base_y-side_length:base_y+side_length, base_x-side_length:base_x+side_length]


            pathing(minimap)

            #screen = cv2.resize(screen, (960,540))
            #cv2.imshow("cv2screen", screen)
            #cv2.waitKey(10)
        cv2.destroyAllWindows()

    elif mode == "screen":
        for i in range(100000):
            screen = grab_screen()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB) # because default will be BGR
            screen = resize(screen)
            #screen = ~screen
            cv2.imshow("cv2screen", screen)
            cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main("speed")