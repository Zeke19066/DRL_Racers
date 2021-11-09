import Monitor_GUI
import Neural_Network
import multiprocessing
import time


if __name__ == "__main__":
    mode = 'train'
    send_connection, recieve_connection = multiprocessing.Pipe()
    gui_process = multiprocessing.Process(target=Monitor_GUI.main, args=(recieve_connection,))
    nn_process = multiprocessing.Process(target=Neural_Network.main, args=(mode, send_connection))

    nn_process.start()
    gui_process.start()

    while 1:
        time.sleep(1)