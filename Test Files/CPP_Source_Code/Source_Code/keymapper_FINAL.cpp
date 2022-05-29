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
void togglePause(bool);

int main(){

    Sleep(3000); // Pause for 3 seconds.
  
    
    auto start = high_resolution_clock::now(); //Begin timing
    int randLast = 2; //initialized
    
    //*firstRun();
    bool mode = true;
    togglePause(mode);
    Sleep(1000);
    mode = false;
    togglePause(mode);
    //*/

    for (int i=0; i < 25; i++) {
        int iRand = rand() % 5;
        //int iRand = 0;
        cout << "Iteration:" << i << "   Action:" << iRand << "\n";
        actionMap(iRand, randLast);
        randLast = iRand;
        //Sleep(100);
        Sleep(10000);
    }

    auto stop = high_resolution_clock::now(); //End timing
    auto duration = duration_cast<microseconds>(stop - start); //Measure Time difference
    cout << "Time taken by function: "
        << duration.count() << " microseconds" << endl; 
    return 0;
}

// Executes agent actions.
void actionMap(int action, int lastAction){
    // This structure will be used to create the keyboard
    INPUT ipActionMap;// This structure will be used to create the keyboard input event.
    ipActionMap.type = INPUT_KEYBOARD;
    ipActionMap.ki.time = 0;
    ipActionMap.ki.wVk = 0; //We're doing scan codes instead
    ipActionMap.ki.dwExtraInfo = 0;

    //int moveKeys[5] = { 0x57, 0x51, 0x53,  0x41, 0x44 }; //WQSAD keys in virtual
    int moveKeys[5] = { 0x11, 0x10, 0x1F,  0x1E, 0x20 }; //WQSAD keys in scancode

    //First remove lastAction
    ipActionMap.ki.wScan = moveKeys[lastAction]; //Select last key
    ipActionMap.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP; // Key-up
    SendInput(1, &ipActionMap, sizeof(INPUT));
    //Always start new action-sequence by going Forwards
    ipActionMap.ki.wScan = moveKeys[0]; //Select W
    ipActionMap.ki.dwFlags = KEYEVENTF_SCANCODE;// Key-down
    ipActionMap.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ipActionMap, sizeof(INPUT));

    if (action == 0)//Forward
        return;
    if (action == 1){//Power-Up
        ipActionMap.ki.wScan = moveKeys[1]; //Select Q
        ipActionMap.ki.dwFlags = 0; // 0 for key press
        SendInput(1, &ipActionMap, sizeof(INPUT));
    }
    if (action == 2){//Reverse
        //first remove the W
        ipActionMap.ki.wScan = moveKeys[0]; //Select W
        ipActionMap.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP; // Key-up
        SendInput(1, &ipActionMap, sizeof(INPUT));

        //Then press down S
        ipActionMap.ki.wScan = moveKeys[2]; //Select S
        ipActionMap.ki.dwFlags = KEYEVENTF_SCANCODE;// Key-down
        ipActionMap.ki.dwFlags = 0; // 0 for key press
        SendInput(1, &ipActionMap, sizeof(INPUT));
    }
    if (action == 3){//Left
        ipActionMap.ki.wScan = moveKeys[3]; //Select A
        ipActionMap.ki.dwFlags = 0; // 0 for key press
        SendInput(1, &ipActionMap, sizeof(INPUT));
    }
    if (action == 4){//Right
        ipActionMap.ki.wScan = moveKeys[4]; //Select D
        ipActionMap.ki.dwFlags = 0; // 0 for key press
        SendInput(1, &ipActionMap, sizeof(INPUT));
    }

}

// Quick reset between races. Causes glithes if game is not quitReset() once every 10 races.
void reset(){
    //Release W
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

// Resets from main menu to prevent overflow glitches (rivals dont use powerups, etc).
void quitReset(){
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
    Sleep(150);

    //press and release DOWN (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(150);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(150);

    //press and release ENTER
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(150);

    //press and release UP
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x26, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(150);

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
    Sleep(150);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY;
    ipReset.ki.wScan = MapVirtualKeyA(0x28, MAPVK_VK_TO_VSC);
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(150);

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
    Sleep(150);

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
    Sleep(150);

    //press and release ENTER (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(150);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(6000);

    //press and release "M" (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(5);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(5);
}

// First run from main menu. First minmap reset different from quitReset().
void firstRun(){
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
    Sleep(300);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x1c; //ENTER
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(6000);

    //press and release "M" (2x)
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(5);
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x32; //M
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(5);

    //press and release "V"
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
    ipReset.ki.wScan = 0x2F;
    SendInput(1, &ipReset, sizeof(INPUT));
    ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ipReset, sizeof(INPUT));
    Sleep(5);
}

// Toggles between Mini-Map and Speedguage based on current mode.
void toggleMode(int current_mode){
    // Set up a generic keyboard event.
    INPUT ipReset;// This structure will be used to create the keyboard input event.
    ipReset.type = INPUT_KEYBOARD;
    ipReset.ki.time = 0;
    ipReset.ki.wVk = 0; //We're doing scan codes instead
    ipReset.ki.dwExtraInfo = 0;

    int pause_time = 2;
    if (current_mode == 0)//Minmap2Speedgauge
    {
        //press and release "M" (3x)
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
        ipReset.ki.wScan = 0x32; //M
        SendInput(1, &ipReset, sizeof(INPUT));
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipReset, sizeof(INPUT));
        Sleep(pause_time);
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
        ipReset.ki.wScan = 0x32; //M
        SendInput(1, &ipReset, sizeof(INPUT));
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipReset, sizeof(INPUT));
        Sleep(pause_time);
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
        ipReset.ki.wScan = 0x32; //M
        SendInput(1, &ipReset, sizeof(INPUT));
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipReset, sizeof(INPUT));
        Sleep(pause_time);
    }
    if (current_mode == 1)//Speedgauge2Minmap
    {
        //press and release "M" (1x)
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
        ipReset.ki.wScan = 0x32; //M
        SendInput(1, &ipReset, sizeof(INPUT));
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipReset, sizeof(INPUT));
        Sleep(pause_time);
    }
    
}

// Toggles pause screen based on bool. True=Pause, False=Unpause
void togglePause(bool mode){
    INPUT ipReset;// This structure will be used to create the keyboard input event.
    ipReset.type = INPUT_KEYBOARD;
    ipReset.ki.time = 0;
    ipReset.ki.wVk = 0; //We're doing scan codes instead
    ipReset.ki.dwExtraInfo = 0;
    
    if (mode == true){ // Pause Game
        //press and release Esc

        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
        ipReset.ki.wScan = 0x01; //ESC
        SendInput(1, &ipReset, sizeof(INPUT));
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipReset, sizeof(INPUT));
    }

    else if (mode == false){ // Un-Pause Game
        //press and release ENTER

        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
        ipReset.ki.wScan = 0x1c; //ENTER
        SendInput(1, &ipReset, sizeof(INPUT));
        ipReset.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipReset, sizeof(INPUT));

        Sleep(300);
        //ipReset.ki.wScan = 0x11; //Select W
        //ipReset.ki.dwFlags = KEYEVENTF_SCANCODE;
        //SendInput(1, &ipReset, sizeof(INPUT));
    }
    return;
}