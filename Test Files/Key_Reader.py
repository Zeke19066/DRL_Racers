import keyboard  # using module keyboard
while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('w'):  # Forward
            if not keyboard.is_pressed('a') and not keyboard.is_pressed('d'):
                print('Foward!')

        if keyboard.is_pressed('q'):  # Fire Powerup
            print('Fire Powerup!')

        if keyboard.is_pressed('a'):  # Left
            print('Left!')

        if keyboard.is_pressed('d'):  # Right
            print('Right!')

        if keyboard.is_pressed('s'):  # Reverse
            print('Reverse!')

    except:
        pass
        #break  # if user pressed a key other than the given key the loop will break
