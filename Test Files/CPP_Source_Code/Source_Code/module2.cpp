
#include <pybind11/pybind11.h>
#include <windows.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#define WINVER 0x0500

using namespace std;

struct actionMap
{
    int action, lastAction;


    actionMap(int action, int lastAction) : action(action), lastAction(lastAction)
    {
        // This structure will be used to create the keyboard
        // input event.
        INPUT ip;

        // Set up a generic keyboard event.
        ip.type = INPUT_KEYBOARD;
        ip.ki.wScan = 0; // hardware scan code for key
        ip.ki.time = 0;
        ip.ki.dwExtraInfo = 0;

        int moveKeys[5] = { 0x57, 0x51, 0x53,  0x41, 0x44 }; //WQSAD keys

        if (lastAction == action)
        {
            return;
        }
        else if (lastAction != action)
        {
            //First remove lastAction
            ip.ki.wVk = moveKeys[lastAction]; //Select last key
            ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
            SendInput(1, &ip, sizeof(INPUT));
            //Always start new action-sequence by going Forwards
            ip.ki.wVk = moveKeys[0]; //Select W
            ip.ki.dwFlags = 0; // 0 for key press
            SendInput(1, &ip, sizeof(INPUT));

            if (action == 0)//Forward
                return;
            if (action == 1)//Power-Up
            {
                ip.ki.wVk = moveKeys[1]; //Select Q
                ip.ki.dwFlags = 0; // 0 for key press
                SendInput(1, &ip, sizeof(INPUT));
            }
            if (action == 2)//Reverse
            {   //first remove the W
                ip.ki.wVk = moveKeys[0]; //Select W
                ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
                SendInput(1, &ip, sizeof(INPUT));

                //Then press down S
                ip.ki.wVk = moveKeys[2]; //Select S
                ip.ki.dwFlags = 0; // 0 for key press
                SendInput(1, &ip, sizeof(INPUT));
            }
            if (action == 3)//Left
            {
                ip.ki.wVk = moveKeys[1]; //Select A
                ip.ki.dwFlags = 0; // 0 for key press
                SendInput(1, &ip, sizeof(INPUT));
            }
            if (action == 4)//Right
            {
                ip.ki.wVk = moveKeys[4]; //Select D
                ip.ki.dwFlags = 0; // 0 for key press
                SendInput(1, &ip, sizeof(INPUT));
            }
        }

    }

};

PYBIND11_MODULE (pybind11module, module)
{
    module.doc () = "pybind11module";

    pybind11::class_<actionMap>(module, "Action Map")
        .def(pybind11::init<>())
        .def(pybind11::init<int, int>(), "constructor 2", pybind11::arg("action"), pybind11::arg("lastAction"))
        ;
}