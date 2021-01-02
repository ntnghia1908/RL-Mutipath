import matplotlib.pyplot as plt
import pickle

BITRATE_TRACE = "bitrate_list_fix"
bitrate_list = pickle.load(open(BITRATE_TRACE,'rb'))
plt.plot(bitrate_list/1024*8)
plt.title("Network trace")
plt.ylabel("download bitrate(kbit)")
plt.show()
