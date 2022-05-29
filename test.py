from collections import deque

que = deque(maxlen=25)

print(type(que))

if str(type(que)) == "<class 'collections.deque'>":
    print("DING")