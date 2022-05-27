import Racers_Train as Racers_Train
import Dataset_Generator

import multiprocessing
import os
import time

#Note: Ensure DXWnd wrapper is open and calibrated correctly before launching.

#Modes = "Generate", "Train", "Train_Supervised", "Test"
mode = "Train"

def main(mode):
    print(f"Pre-Startup Mode:{mode};")
    if mode == "Generate":
        #start the game
        racers_shortcut_dir = r"Resources\LEGORacers.lnk"
        racers = os.startfile(racers_shortcut_dir)
        time.sleep(10)

        dataset = Dataset_Generator.Dataset_Generator()
        while 1:
            dataset.Generator()

    elif mode == "Train":
        import Racers_GUI
        mode = 'train'
        #start the game
        
        racers_shortcut_dir = r"Resources\LEGORacers.lnk"
        racers = os.startfile(racers_shortcut_dir)
        time.sleep(5)

        #Initialize multiprocess pipeline.
        send_connection, recieve_connection = multiprocessing.Pipe()
        gui_process = multiprocessing.Process(target=Racers_GUI.main, args=(recieve_connection,)) #gotta leave that comma
        nn_process = multiprocessing.Process(target=Racers_Train.main, args=(mode, send_connection))

        nn_process.start()
        gui_process.start()

        while 1:
            time.sleep(1)

    elif mode == "Train_Supervised":
        import Racers_GUI_Show_Me as Racers_GUI
        mode = 'train_supervised'

        send_connection, recieve_connection = multiprocessing.Pipe()
        gui_process = multiprocessing.Process(target=Racers_GUI.main, args=(recieve_connection,)) #gotta leave that comma
        nn_process = multiprocessing.Process(target=Racers_Train.main, args=(mode, send_connection))

        nn_process.start()
        gui_process.start()

        while 1:
            time.sleep(1)

    elif mode == "Test":
        import Racers_GUI
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