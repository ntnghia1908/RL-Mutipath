import requests
import numpy as np
import pickle

class video_list_collector():

    def __init__(self,
                base_link = "http://ftp.itec.aau.at/datasets/DASHDataset2014/BigBuckBunny/1sec/",
                form = "bunny_{0}bps/BigBuckBunny_1s{1}.m4s",
                save_dir = "video_list"):
        self.base_link = base_link
        self.form = form
        self.save_dir = save_dir

    def seperate_trace(self,available_bitrate = [46980, 91917, 135410, 
                    182366, 226106, 270316, 352546, 620705, 808057, 1071529, 
                    1312787, 1662809, 2234145, 2617284, 3305118,3841983, 4242923, 
                    4726737
                    ]):
        self.available_bitrate = available_bitrate
        self.segment_trace = {}
        for x in available_bitrate:
            seg_size = []
            seg_num = 1
            while(True):
                link = self.base_link + self.form.format(str(x),str(seg_num))
                r = requests.head(link)
                try:
                    seg_size.append(r.headers['Content-Length'])
                    seg_num += 1
                except:
                    break
            if seg_num > 2:
                self.segment_trace['{0}'.format(str(x))] = seg_size
                print('collect for {}bps trace completed'.format(str(x)))
            else:
                print('video have no bitrate level {0}'.format(str(x)))

    def get_trace_matrix(self, bitrate_list):
        return_matrix = []
        for bitrate in bitrate_list:
            try:
                return_matrix.append(self.segment_trace[str(bitrate)])
            except:
                print('trace for bitrate {0}bps does not exist'.format(str(bitrate)))
        
        return np.asarray(return_matrix,dtype = np.float32)

    def save(self):
        pickle.dump(self.segment_trace, open(self.save_dir, 'wb'))

    def load(self):
        self.segment_trace = pickle.load(open(self.save_dir, 'rb'))
        print('available bitrate level')
        for k,v in self.segment_trace.items():
            print('{0}bps'.format(k))