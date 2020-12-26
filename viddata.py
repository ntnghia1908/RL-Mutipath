import matplotlib.pyplot as plt
import pickle
from get_down_size import video_list_collector

VIDEO_TRACE = "video_list"
VIDEO_BIT_RATE = [300,700,1200,1500,3000,4000,5000]

vlc = video_list_collector()
vlc.save_dir = VIDEO_TRACE
vlc.load()
video_list = vlc.get_trace_matrix(VIDEO_BIT_RATE)
data = []
for trace in video_list:
    data.append(trace/1024*8)

fig, ax = plt.subplots()
ax.set_title('Video data')
ax.set_xticklabels(VIDEO_BIT_RATE)
ax.boxplot(data, showfliers=False)
plt.ylabel("segment size(kbit)")
plt.xlabel("quality")
plt.show()
