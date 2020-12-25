import pickle
import matplotlib.pyplot as plt
from numpy import array, random
import pickle

C = array([0.25, 0.5, 1, 2, 3, 4, 6, 10])

cur_bitrate = random.choice(C)
bitrate_list = [cur_bitrate]
i = random.rand()

for j in range(1000):
  if i < 0.5:
    cur_bitrate = random.choice(C)
  bitrate_list.append(cur_bitrate)

pickle.dump(bitrate_list, open('markovian_bitrate', 'wb'))

plt.plot(bitrate_list)
plt.xlabel('t')
plt.ylabel('bitrate')
plt.show()
