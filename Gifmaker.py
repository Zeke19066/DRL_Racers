from PIL import Image
import cv2
import os
from decorators import function_timer
import time
import numpy as np
from Racers_Wrapper import ScreenGrab
"""
#To generate frame in pygame:
frame = pygame.surfarray.array3d(self.dis)
self.gif.snap_maker(frame)

#Milestones we wish to capture, in order:
    Any time reward performance is >= 1% better than last high score.
    First time we complete lap1, lap2, and first time finishing the race.
    Every 25th race.
    Every time we finish the race.
"""

class Capture():
    def __init__(self):
        print(os.getcwd())
        self.parent_dir = r"Demo"
        os.chdir(self.parent_dir)
        self.game_count = 0
        self.gif_stack = []
        self.fps_target = 30
        self.lap_last = 1

    def main_loop(self):
        grab = ScreenGrab()
        print("Starting ScreenGrab")
        @function_timer
        def screengrab_subfunction():
            for _ in range(500):
                img = grab.quick_Grab()
                lap_crop = img[21:72, 632:706]
                lap = grab.lap_scan(lap_crop)
                self.record_frame(img)
                print(lap)
                if int(lap) != self.lap_last :
                    print("Lap Over, saving!")
                    self.save_gif()
                    return
            self.save_gif()
            return
        
        screengrab_subfunction()

    #Provide captured frame
    def record_frame(self, img_array):
        img_array = cv2.resize(img_array, (500, 500))
        PIL_image = Image.fromarray(np.uint8(img_array)).convert('RGB')
        self.gif_stack.append(PIL_image)

    def save_gif(self, name="None"):
        self.game_count += 1
        if name == "None":
            name = f'{self.game_count}out.gif'

        self.gif_stack[0].save(name, save_all=True,
                                append_images=self.gif_stack[1:],
                                optimize=True, duration=100, loop=0)
        self.gif_stack = []

def main():
    cap = Capture()
    cap.main_loop()

if __name__ == "__main__":
    main()
