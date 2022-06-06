from cv2 import Mat_MAGIC_VAL
import Racers_Train as Racers_Train
import multiprocessing
import os
import time

#Note: Ensure DXWnd wrapper is open and calibrated correctly before launching.

#Modes = "Train", "Train_Supervised", "Test"
mode = "Train_Supervised"

def main(mode):
    print(f"Pre-Startup Mode:{mode}", end="")

    if mode == "Train":
        import Racers_GUI
        mode = 'train'

        #Initialize multiprocess pipeline.
        send_connection, recieve_connection = multiprocessing.Pipe()
        gui_process = multiprocessing.Process(target=Racers_GUI.main, args=(recieve_connection,)) #gotta leave that comma
        nn_process = multiprocessing.Process(target=Racers_Train.main, args=(mode, send_connection))

        nn_process.start()
        gui_process.start()

        while 1:
            time.sleep(1)

    elif mode == "Train_Supervised":
        import Racers_GUI_Supervised as Racers_GUI
        mode = 'train_supervised'

        send_connection, recieve_connection = multiprocessing.Pipe()
        gui_process = multiprocessing.Process(target=Racers_GUI.main, args=(recieve_connection,)) #gotta leave that comma
        nn_process = multiprocessing.Process(target=Racers_Train.main, args=(mode, send_connection))

        nn_process.start()
        gui_process.start()

        while 1:
            time.sleep(1)

    elif mode == "Test":
    
        mode = 'test'
        #start the game
        
        racers_shortcut_dir = r"Resources\LEGORacers.lnk"
        racers = os.startfile(racers_shortcut_dir)
        time.sleep(7)

        #Initialize multiprocess pipeline.
        send_connection, recieve_connection = multiprocessing.Pipe()
        gui_process = multiprocessing.Process(target=Racers_GUI.main, args=(recieve_connection,)) #gotta leave that comma
        nn_process = multiprocessing.Process(target=Racers_Train.main, args=(mode, send_connection))

        nn_process.start()
        gui_process.start()

        while 1:
            time.sleep(1)

if __name__ == "__main__":
    main(mode)
