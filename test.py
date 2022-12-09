import numpy as np
from collections import deque

placements = deque(maxlen=5)
placements.append([20,0,0,30,40])
#placements.append([20,0,0,30,50])
#placements.append([10,0,0,40,20])

temp_dist = np.array(placements)
move_dist = temp_dist.transpose()

total = np.sum(move_dist)
print(move_dist)
length = len(move_dist)

output = [0 for _ in range(length)]

for i in range(length):
    output[i] = round(np.sum(move_dist[i])/total,3)

print(output)

"""
For example: we have 12% of the moves R, but we want 56% of the moves as R

"""