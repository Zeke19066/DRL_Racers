from decorators import function_timer
import numpy as np
import cv2
import mss

print("Calculating Capture-Rate(60 sec).........",end="")
@function_timer
def screengrab_subfunction():
    for i in range(1000):
        with mss.mss() as sct:
            monitor = {"top": 389, "left": 846, "width": 305, "height": 352}
            img_array = np.array(sct.grab(monitor))
            #status = cv2.imwrite("mini_img_sample.png", img_array)
            #cv2.imshow("cv2screen", img_array)
            #cv2.waitKey(1)

screengrab_subfunction()
#60fps