from get_down_size import video_list_collector
from get_network_data import network_data_getter

BASELINK = "http://ftp.itec.aau.at/datasets/mmsys12/ElephantsDream/ed_4s/"
FOLDER = "pedestrian"
FILE = "A_2017.11.21_15.03.50"

reload_trace = True

vlc = video_list_collector()
ndg = network_data_getter()
if reload_trace:
    ndg.get_trace(FOLDER, FILE)
    vlc.seperate_trace(baselink = BASELINK)
    vlc.save()


vlc.load()
