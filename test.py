import numpy as np
from collections import deque

placements = deque(maxlen=5)
placements.append([20,30,40])
placements.append([20,30,50])
placements.append([10,40,20])

place = np.array(placements)
new_place = place.transpose()
print(new_place)