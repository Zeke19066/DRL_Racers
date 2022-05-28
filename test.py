from collections import deque

que = deque(maxlen=25)

que.append(9)
que.append(9)
que.append(9)
que.append(8)
#que.pop()
que.append(9)
print(que)