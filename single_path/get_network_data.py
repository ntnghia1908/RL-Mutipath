import os
import csv
import pickle
import pandas as pd

class network_data_getter:
    def __init__(self, save_dir = 'bitrate_list'):
        pass
        self.save_dir = save_dir

    def get_trace(self, transportation = "bus", time = "A_2017.11.30_16.48.26"):
        basepath = 'Network_Data/{0}'.format(transportation)
        i = 0
        network_data = pd.read_csv('{0}/{1}.csv'.format(basepath,time))
        network_data = network_data["DL_bitrate"]*1024/8
        pickle.dump(network_data,open(self.save_dir,"wb"))
        return network_data

        