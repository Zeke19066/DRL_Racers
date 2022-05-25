#define WINVER 0x0500
#include <windows.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include "keymapper.h" // User defined headers use quotation marks

using namespace std;

void actionMap(int action, int lastAction)
{
    // This structure will be used to create the keyboard
    // input event.
    INPUT ip;

    // Set up a generic keyboard event.
    ip.type = INPUT_KEYBOARD;
    ip.ki.wScan = 0; // hardware scan code for key
    ip.ki.time = 0;
    ip.ki.dwExtraInfo = 0;

    int moveKeys[5] = {0x57, 0x51, 0x53,  0x41, 0x44}; //WQSAD keys

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

void reset()
{

    // This structure will be used to create the keyboard
    // input event.
    INPUT ip;

    // Set up a generic keyboard event.
    ip.type = INPUT_KEYBOARD;
    ip.ki.wScan = 0; // hardware scan code for key
    ip.ki.time = 0;
    ip.ki.dwExtraInfo = 0;

    //First we lift off W
    ip.ki.wVk = 0x57; //Select W
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));
    //press and release Esc
    ip.ki.wVk = 0x1B; // virtual-key code for Esc
    ip.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ip, sizeof(INPUT));
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));
    Sleep(250);
    //press and release DOWN
    ip.ki.wVk = 0x28; // virtual-key code for DOWN
    ip.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ip, sizeof(INPUT));
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));
    Sleep(250);
    //press and release ENTER
    ip.ki.wVk = 0x28; // virtual-key code for DOWN
    ip.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ip, sizeof(INPUT));
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));
    Sleep(250);
    //Press W (NO RELEASE)
    ip.ki.wVk = 0x57; // virtual-key code for W
    ip.ki.dwFlags = 0; // 0 for key press
    Sleep(5000);
    SendInput(1, &ip, sizeof(INPUT));
    return;
}

void quitReset()
{
    // Set up a generic keyboard event.
    INPUT ipReset;// This structure will be used to create the keyboard input event.
    ipReset.type = INPUT_KEYBOARD;
    ipReset.ki.time = 0;
    ipReset.ki.wVk = 0; //We're doing scan codes instead
    ipReset.ki.dwExtraInfo = 0;

    //press and release Esc
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x01; //ESC
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release DOWN (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release UP
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x26, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(300);

    //press and release DOWN (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(300);

    //press and release UP
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x26, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release RIGHT (3x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x27, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(500);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x27, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(500);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x27, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(500);

    //press and release DOWN
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(10000);

    //press and release "M" (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
}

void firstRun()
{
    // Set up a generic keyboard event.
    INPUT ipReset;// This structure will be used to create the keyboard input event.
    ipReset.type = INPUT_KEYBOARD;
    ipReset.ki.time = 0;
    ipReset.ki.wVk = 0; //We're doing scan codes instead
    ipReset.ki.dwExtraInfo = 0;

    //press and release DOWN (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(300);

    //press and release UP
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x26, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release RIGHT (3x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x27, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(500);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x27, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(500);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x27, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(500);

    //press and release DOWN
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(10000);

    //press and release "M" (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release "V"
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x2F;
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);
}

