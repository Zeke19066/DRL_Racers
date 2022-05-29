// keystroke.c - Pauses, then simulates a key press
// and release of the "A" key.
// ...then switch to e.g. a Notepad window and wait
// 5 seconds for the A key to be magically pressed.
// Because the SendInput function is only supported in
// Windows 2000 and later, WINVER needs to be set as
// follows so that SendInput gets defined when windows.h
// is included below.
// W:0x57;  A:0x41:  S:0x53:  D:0x44;  Q:0x51

//#define WINVER 0x0500
#include <windows.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <chrono>       //For timing a function

using namespace std;
using namespace std::chrono; //For timing a function

void actionMap(int, int);
void reset();
void ShowDesktop();
void dPresser();
void quitReset();
void firstRun();

int main()
{
    Sleep(3000); // Pause for 3 seconds.
    int i = 0;
    int randLast = 2; //initialized out of range.
    auto start = high_resolution_clock::now(); //Begin timing
    
    firstRun();

    /* v2
    while (i < 50) //NEW
    {
        cout << "Iteration:" << i << "\n";
        i++;
        int iRand = rand() % 5;

        if(iRand == randLast)
        {
            cout << "REPEAT" << iRand << "\n";
        }

        else if(iRand != randLast)
        {

            //First remove lastAction
            actionMap(randLast, 1); //lift last key
            randLast = iRand;

            //Start with a W
            actionMap(0, 0); //press w key

            if (iRand == 0)//Forward
            {
                cout << "Forward" << iRand << "\n";
            }
            if (iRand == 1)//Power-Up
            {
                actionMap(1, 0); //press q key
            }
            if (iRand == 2)//Reverse
            {   
                //first remove the W
                actionMap(0, 1); //lift w key

                //Then press down S
                actionMap(2, 0); //press s key
            }
            if (iRand == 3)//Left
            {
                actionMap(3, 0); //press a key
            }
            if (iRand == 4)//Right
            {
                actionMap(4, 0); //press d key
            }
        }
        Sleep(500);
    }
    

    //cLASSIC
    while (i < 25) 
    {
        cout << "Iteration:" << i << "\n";
        i++;
        int iRand = rand() % 5;
        actionMap(iRand, randLast);
        randLast = iRand;
        Sleep(100);
    }
    */
    
    auto stop = high_resolution_clock::now(); //End timing
    auto duration = duration_cast<microseconds>(stop - start); //Measure Time difference
    cout << "Time taken by function: "
        << duration.count() << " microseconds" << endl; 
    return 0;
}

void actionMapv2(int action, int direction) // Direction keydown:0 keyup:1
{
    int moveKeys[5] = {0x11, 0x10, 0x1f,  0x1e, 0x20}; //WQSAD keys

    if (direction == 0)// keydown
    {
        INPUT ipUp; // Set up a generic keyboard event.
        ipUp.type = INPUT_KEYBOARD;
        ipUp.ki.wVk = 0; //We're doing scan codes instead
        ipUp.ki.time = 0;
        ipUp.ki.dwExtraInfo = 0;

        ipUp.ki.wScan = moveKeys[action]; //Select W
        ipUp.ki.dwFlags = KEYEVENTF_SCANCODE;
        SendInput(1, &ipUp, sizeof(INPUT));
        return;
    }

    else if (direction == 1)// keyup
    {
        INPUT ipDown; // Set up a generic keyboard event.
        ipDown.type = INPUT_KEYBOARD;
        ipDown.ki.wVk = 0; //We're doing scan codes instead
        ipDown.ki.time = 0;
        ipDown.ki.dwExtraInfo = 0;

        ipDown.ki.wScan = moveKeys[action]; //Select last key
        ipDown.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipDown, sizeof(INPUT));
        return;
    }
};

void actionMap(int action, int lastAction)
{
    int moveKeys[5] = {0x11, 0x10, 0x1f,  0x1e, 0x20}; //WQSAD keys

    if (lastAction == action)
    {
        return;
    }

    else if (lastAction != action)
    {

        // This structure will be used to create the keyboard
        // input event.
        INPUT ipAct; // Set up a generic keyboard event.
        ipAct.type = INPUT_KEYBOARD;
        ipAct.ki.wVk = 0; //We're doing scan codes instead
        ipAct.ki.time = 0;
        ipAct.ki.dwExtraInfo = 0;

        //First remove lastAction
        ipAct.ki.wScan = moveKeys[lastAction]; //Select last key
        ipAct.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipAct, sizeof(INPUT));

        //Start with a W
        ipAct.ki.wScan = moveKeys[0]; //Select W
        ipAct.ki.dwFlags = KEYEVENTF_SCANCODE;
        SendInput(1, &ipAct, sizeof(INPUT));

        if (action == 0)//Forward
        {
            return;
        }
        if (action == 1)//Power-Up
        {
            ipAct.ki.wScan = moveKeys[1]; //Select Q
            ipAct.ki.dwFlags = KEYEVENTF_SCANCODE;
            SendInput(1, &ipAct, sizeof(INPUT));
            return;
        }
        if (action == 2)//Reverse
        {   //first remove the W
            ipAct.ki.wScan = moveKeys[0]; //Select W
            ipAct.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
            SendInput(1, &ipAct, sizeof(INPUT));

            //Then press down S
            ipAct.ki.wScan = moveKeys[2]; //Select S
            ipAct.ki.dwFlags = KEYEVENTF_SCANCODE;
            SendInput(1, &ipAct, sizeof(INPUT));
            return;
        }
        if (action == 3)//Left
        {
            ipAct.ki.wScan = moveKeys[3]; //Select A
            ipAct.ki.dwFlags = KEYEVENTF_SCANCODE;
            SendInput(1, &ipAct, sizeof(INPUT));
            return;
        }
        if (action == 4)//Right
        {
            ipAct.ki.wScan = moveKeys[4]; //Select D
            ipAct.ki.dwFlags = KEYEVENTF_SCANCODE;
            SendInput(1, &ipAct, sizeof(INPUT));
            return;
        }
        return;
    }
};

void reset_v2()
{
    //Release W
    INPUT ipReset_1;// This structure will be used to create the keyboard input event.
    ipReset_1.type = INPUT_KEYBOARD;
    ipReset_1.ki.time = 0;
    ipReset_1.ki.wVk = 0; //We're doing scan codes instead
    ipReset_1.ki.dwExtraInfo = 0;

    ipReset_1.ki.wScan = 0x11; //Select W
    ipReset_1.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset_1, sizeof(INPUT));

    //press and release Esc
    INPUT ipReset_2;// This structure will be used to create the keyboard input event.
    ipReset_2.type = INPUT_KEYBOARD;
    ipReset_2.ki.time = 0;
    ipReset_2.ki.wVk = 0; //We're doing scan codes instead
    ipReset_2.ki.dwExtraInfo = 0;

    ipReset_2.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset_2.ki.wScan = 0x01; //ESC
    SendInput(1, &ipReset_2, sizeof(INPUT));
    ipReset_2.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset_2, sizeof(INPUT));
    Sleep(100);

    //press and release DOWN
    INPUT ipReset_3;// This structure will be used to create the keyboard input event.
    ipReset_3.type = INPUT_KEYBOARD;
    ipReset_3.ki.time = 0;
    ipReset_3.ki.wVk = 0; //We're doing scan codes instead
    ipReset_3.ki.dwExtraInfo = 0;

    ipReset_3.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset_3.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset_3, sizeof(INPUT));
    ipReset_3.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset_3, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER
    INPUT ipReset_4;// This structure will be used to create the keyboard input event.
    ipReset_4.type = INPUT_KEYBOARD;
    ipReset_4.ki.time = 0;
    ipReset_4.ki.wVk = 0; //We're doing scan codes instead
    ipReset_4.ki.dwExtraInfo = 0;

    ipReset_4.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset_4.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset_4, sizeof(INPUT));
    ipReset_4.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset_4, sizeof(INPUT));
    Sleep(100);

    //press and release UP
    INPUT ipReset_5;// This structure will be used to create the keyboard input event.
    ipReset_5.type = INPUT_KEYBOARD;
    ipReset_5.ki.time = 0;
    ipReset_5.ki.wVk = 0; //We're doing scan codes instead
    ipReset_5.ki.dwExtraInfo = 0;

    ipReset_5.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset_5.ki.wScan = MapVirtualKeyA(0x26, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset_5, sizeof(INPUT));
    ipReset_5.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset_5, sizeof(INPUT));
    Sleep(100);

    //press and release ENTER
    INPUT ipReset_6;// This structure will be used to create the keyboard input event.
    ipReset_6.type = INPUT_KEYBOARD;
    ipReset_6.ki.time = 0;
    ipReset_6.ki.wVk = 0; //We're doing scan codes instead
    ipReset_6.ki.dwExtraInfo = 0;

    ipReset_6.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset_6.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset_6, sizeof(INPUT));
    ipReset_6.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset_6, sizeof(INPUT));
    Sleep(100);

    //Press W
    INPUT ipReset_7;// This structure will be used to create the keyboard input event.
    ipReset_7.type = INPUT_KEYBOARD;
    ipReset_7.ki.time = 0;
    ipReset_7.ki.wVk = 0; //We're doing scan codes instead
    ipReset_7.ki.dwExtraInfo = 0;

    ipReset_7.ki.wScan = 0x11; //Select W
    ipReset_7.ki.dwFlags = KEYEVENTF_SCANCODE;
    SendInput(1, &ipReset_7, sizeof(INPUT));
    return;
}

void reset()
{
    // Set up a generic keyboard event.
    INPUT ipReset;// This structure will be used to create the keyboard input event.
    ipReset.type = INPUT_KEYBOARD;
    ipReset.ki.time = 0;
    ipReset.ki.wVk = 0; //We're doing scan codes instead
    ipReset.ki.dwExtraInfo = 0;

    //Release W
    ipReset.ki.wScan = 0x11; //Select W
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));

    //press and release Esc
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x01; //ESC
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(100);

    //press and release DOWN
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
    Sleep(100);

    //Press W
    ipReset.ki.wScan = 0x11; //Select W
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    SendInput(1, &ipReset, sizeof(INPUT));

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
