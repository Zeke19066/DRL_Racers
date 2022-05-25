{
    cout << "Starting ActionMap" << "\n";
    int moveKeys[5] = {0x11, 0x10, 0x1f,  0x1e, 0x20}; //WQSAD keys

    if (direction == 0)// keydown
    {
        cout << "KEY DOWN" << "\n";
        INPUT ipUp; // Set up a generic keyboard event.
        ipUp.type = INPUT_KEYBOARD;
        ipUp.ki.wVk = 0; //We're doing scan codes instead
        ipUp.ki.time = 0;
        ipUp.ki.dwExtraInfo = 0;

        ipUp.ki.wScan = moveKeys[action]; //Select W
        ipUp.ki.dwFlags = KEYEVENTF_SCANCODE;
        SendInput(1, &ipUp, sizeof(INPUT));
        cout << "DONE KEY DOWN" << endl;
        return;
    }

    else if (direction == 1)// keyup
    {
        cout << "KEY UP" << "\n";
        INPUT ipDown; // Set up a generic keyboard event.
        ipDown.type = INPUT_KEYBOARD;
        ipDown.ki.wVk = 0; //We're doing scan codes instead
        ipDown.ki.time = 0;
        ipDown.ki.dwExtraInfo = 0;

        ipDown.ki.wScan = moveKeys[action]; //Select last key
        ipDown.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &ipDown, sizeof(INPUT));
        cout << "DONE KEY UP" << endl;
        return;
    }
};