import Racers_GUI
#import Racers_Train_A2C as Racers_Train
#import Racers_Train_LowGamma as Racers_Train


import Racers_Train_A2C_Lin as Racers_Train

#import Dino_Flav.Racers_GUI_Dino as Racers_GUI
#import Dino_Flav.Racers_Train_Dino as Racers_Train
import multiprocessing
import os
import time

#Note: Ensure DXWnd wrapper is open and calibrated correctly before launching.

if __name__ == "__main__":
    mode = 'train'
    racers_shortcut_dir = r"Resources\LEGORacers.lnk"
    racers = os.startfile(racers_shortcut_dir)
    time.sleep(5)

    send_connection, recieve_connection = multiprocessing.Pipe()
    gui_process = multiprocessing.Process(target=Racers_GUI.main, args=(recieve_connection,)) #gotta leave that comma
    nn_process = multiprocessing.Process(target=Racers_Train.main, args=(mode, send_connection))

    nn_process.start()
    gui_process.start()

    while 1:
        time.sleep(1)