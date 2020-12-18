import pickle
import numpy as np
import copy
from .video_player import VideoPlayer
from .get_down_size import video_list_collector

class Env():
    BITRATE_TRACE = 'simulator/bitrate_list'
    VIDEO_TRACE = 'simulator/video_list_4s'
    A_DIM = 7
    HISTORY_SIZE = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
    TRACE_SIZE = 8  # take how many frames in the past
    STATE_SIZE = (HISTORY_SIZE, TRACE_SIZE)
    BUFFER_NORM_FACTOR = 10.0
    CHUNK_TIL_VIDEO_END_CAP = 60.0
    M_IN_K = 1000.0
    REBUF_PENALTY = 10  # 1 sec rebuffering -> 3 Mbps
    SMOOTH_PENALTY = 1
    DEFAULT_QUALITY = 1  # default video quality without agent
    # video bitrate is used as a ultility reward for each bitrate level so this can be change however fit
    # UTILITY_SCORE = [60,170,340,1010,2080,4130,5900] (original)
    UTILITY_SCORE = [300,700,1200,1500,3000,4000,5000] # for 04-second segment

    # UTILITY_SCORE = [0,1,2,3,4,5,6]
    #VIDEO_BIT_RATE = [46980,135410,270316,808057,1662809,3305118,4726737] # unit is bit, 1-second segment (original)
    VIDEO_BIT_RATE = [300,700,1200,1500,3000,4000,5000] # for 04-second segment

    birate_list = pickle.load(open(BITRATE_TRACE,'rb'))
    # 650000 500000 450000
    # birate_list = [100000000]*10
    # birate_list = np.array([450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,
    # 650000, 650000, 650000, 650000, 650000, 650000, 650000, 650000, 650000, 650000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,
    # 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,])/1.5
    # birate_list = np.array([99999999]*2)
    # birate_list = np.array([100000]*2)
    # birate_list = np.array([424615.25247604, 426075.84716378, 467634.18403817, 440075.95913809,
    #    436539.13034171, 434932.0146163 , 430107.5987599 , 405391.81089049,
    #    484076.28302436, 488954.94536026, 413910.71173884, 483176.94889314,
    #    480011.69770384, 430961.44226432, 410630.82675466, 445046.71992339,
    #    422555.28714071, 446220.29079491, 439991.62569428, 408095.01544452,
    #    461558.8708621 , 441129.85764814, 453185.73665372, 405234.66815795,
    #    429360.21774155, 437702.75337378, 463164.49456149, 449447.15082642,
    #    442870.61085563, 421703.62714853, 407713.85015207, 455975.31342512,
    #    488420.84014968, 456307.63649371, 464204.9359653 , 468104.78356353,
    #    410203.26005775, 461111.7357427 , 416732.33732674, 423662.7783588 ,
    #    435471.22132476, 484322.75175775, 491902.97521319, 418904.93452714,
    #    459105.44744608, 472343.58897459, 493121.66247686, 447069.86917348,
    #    412419.45365055, 419976.13452433, 473994.77149273, 432848.97884417,
    #    456697.05360129, 436486.64587848, 438852.85709929, 470685.88359501,
    #    420751.91028817, 428459.1025539 , 439434.33557232, 428363.9844962 ])
    print('Network bandwidth min {:.2f} mean {:.2f} max {:.2f}'.format(np.min(birate_list), np.mean(birate_list), np.max(birate_list)))

    def __init__(self):
        super().__init__()

        assert len(self.UTILITY_SCORE) == len(self.VIDEO_BIT_RATE)
        # get video list
        vlc = video_list_collector()
        vlc.save_dir = self.VIDEO_TRACE
        vlc.load()
        self.video_list = vlc.get_trace_matrix(self.VIDEO_BIT_RATE)
        #self.video_list = self.video_list[:, 0:60]
        print("Len", len(self.video_list[0]))
        
        self.video_player = VideoPlayer(self.birate_list, self.video_list)
        self.buffer_thresh = self.video_player.buffer_thresh
        # self.action_list = len(self.video_list)
        self.ACTION_SIZE = len(self.video_list)
        self.reset()

    # def state_size(self):
    #     return self.HISTORY_SIZE, self.TRACE_SIZE

    # def action_size(self):
    #     return self.action_list

    def reset(self):
        self.video_player.reset()
        self.last_action = self.DEFAULT_QUALITY
        self.state = np.array(np.zeros((self.HISTORY_SIZE,self.TRACE_SIZE)),dtype=np.float32)
        terminate = False
        reward = 0
        return self.state

    def process_trace(self,data):
        data = copy.deepcopy(data)
        data.pop("next_seg")
        data.pop("segment")
        terminate = data["terminate"]
        data.pop("terminate", None)
        reshape = [v for v in data.values()]
        data = np.array(reshape)
        return data, terminate
    
    def cal_reward(self, action, rebuf):
        reward = self.UTILITY_SCORE[action] \
                    - self.SMOOTH_PENALTY * np.abs(self.UTILITY_SCORE[action] 
                    - self.UTILITY_SCORE[self.last_action]) \
                    - self.REBUF_PENALTY * rebuf

        # reward = (self.UTILITY_SCORE[action]*0.3 + (6 - abs(self.UTILITY_SCORE[action] - self.UTILITY_SCORE[self.last_action]))*0.2)/6
        # if rebuf == 0:
        #     reward += 0.5


        # print('-> Action {}\tReward {:.2f}\tRebuf {:.2f}'.format(action, reward, rebuf))

        return reward

    def step(self,action):
        last_state = self.state
        raw_trace = self.video_player.download(action)
        trace, terminate = self.process_trace(raw_trace)
        
        last_trace = self.state[0]
        self.state = np.roll(self.state,1)
        self.state[0] = trace 

        reward = self.cal_reward(last_trace, trace)
        return self.state.flatten(), reward, terminate, last_state.flatten()

    def step_new(self, action):
        #return delay, sleep_time, return_buffer, rebuf, return_segment, next_segments, terminate, remain 
        last_state = self.state
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, bw = \
            self.video_player.download(action)

        self.state = np.roll(self.state, -1, axis=1)
		
        delay *= 1000 # in milisec

        self.state[0, -1] = self.UTILITY_SCORE[action] / float(np.max(self.UTILITY_SCORE))  # scale from [0,1]
        self.state[1, -1] = buffer_size / self.BUFFER_NORM_FACTOR  # from 0-3 after nomalized
        self.state[2, -1] = float(video_chunk_size) / float(delay) / self.M_IN_K # bps/ms/1000, estimated throughput in Mbps
        self.state[3, -1] = float(delay) / self.M_IN_K  # self.BUFFER_NORM_FACTOR   # delay in sec / 10
        self.state[4, :self.A_DIM] = np.array(next_video_chunk_sizes) / self.M_IN_K / self.M_IN_K  # in chunk size Mbps
        self.state[5, -1] = np.minimum(video_chunk_remain, self.CHUNK_TIL_VIDEO_END_CAP) / float(self.CHUNK_TIL_VIDEO_END_CAP) # scale to [0,1]
        
        reward = self.cal_reward(action, rebuf)
        self.last_action = action

        print(self.state)
        return self.state, reward, end_of_video, buffer_size, video_chunk_size, self.state[2, -1], bw

