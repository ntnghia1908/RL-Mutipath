import pickle
import numpy as np
import copy
from video_player import VideoPlayer
from get_down_size import video_list_collector

class Env():

    BITRATE_TRACE = 'markovian_bitrate'
    VIDEO_TRACE = 'video_list'
    HISTORY_SIZE = 7
    BUFFER_NORM_FACTOR = 10.0
    CHUNK_TIL_VIDEO_END_CAP = 60.0
    M_IN_K = 1000.0
    REBUF_PENALTY = 10  # 1 sec rebuffering -> 3 Mbps
    SMOOTH_PENALTY = 1
    DEFAULT_QUALITY = 1  # default video quality without agent
    # video bitrate is used as a ultility reward for each bitrate level so this can be change however fit
    UTILITY_SCORE = [300,700,1200,1500,3000,4000,5000]
    VIDEO_BIT_RATE = [300,700,1200,1500,3000,4000,5000] # unit is bit
    
    birate_list = pickle.load(open(BITRATE_TRACE,'rb'))

    def __init__(self):
        super().__init__()

        assert len(self.UTILITY_SCORE) == len(self.VIDEO_BIT_RATE)
        # get video list
        vlc = video_list_collector()
        vlc.save_dir = self.VIDEO_TRACE
        vlc.load()
        self.video_list = vlc.get_trace_matrix(self.VIDEO_BIT_RATE)

        self.video_player = VideoPlayer(self.birate_list,self.video_list)
        self.buffer_thresh = self.video_player.buffer_thresh
        self.action_list = len(self.video_list)
        self.network_speed = np.zeros(self.HISTORY_SIZE)
        self.reset()

    def action_space(self):
        return self.action_list

    def state_space(self):
        """
        return: state space of the env
        """
        return self.state.shape

    def reset(self, mode = 'train'):    
        next_video_chunk_size = self.video_player.reset(mode = mode)
        self.last_action = self.DEFAULT_QUALITY
        self.network_speed = np.zeros(self.HISTORY_SIZE)
        self.state = np.append(self.network_speed, 0)       #net speed , buffer
        self.state = np.append(self.state, 0)               #chunk remain
        self.state = np.append(self.state, next_video_chunk_size / 1e6) #next options
        self.state = np.append(self.state, 0)               #last action
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
        reward = self.UTILITY_SCORE[action] / self.M_IN_K \
                    - self.REBUF_PENALTY * rebuf \
                    - self.SMOOTH_PENALTY * np.abs(self.UTILITY_SCORE[action]
                    - self.UTILITY_SCORE[self.last_action]) / self.M_IN_K

        return reward

    def get_sep_reward(self, action, rebuf):
        quality = self.UTILITY_SCORE[action] / self.M_IN_K
        rebuf_pen = self.REBUF_PENALTY * rebuf
        switch_pen = self.SMOOTH_PENALTY * np.abs(self.UTILITY_SCORE[action]
                    - self.UTILITY_SCORE[self.last_action]) / self.M_IN_K

        return quality, rebuf_pen, switch_pen


    def step(self,action, sep = False):
        last_state = self.state
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            self.video_player.download(action)

        reward = self.cal_reward(action, rebuf)
        quality, rebuf_pen, switch_pen = self.get_sep_reward(action, rebuf)
        self.last_action = action
        net_speed = video_chunk_size / (delay - sleep_time)
        self.network_speed = np.roll(self.network_speed,axis = -1,shift = 1)
        self.network_speed[0] = net_speed / 1e6
        self.state = np.append(self.network_speed, buffer_size * 0.1)
        self.state = np.append(self.state, video_chunk_remain * 0.1)
        self.state = np.append(self.state, next_video_chunk_sizes / 1e7)
        self.state = np.append(self.state, self.last_action * 0.1)
        
        return self.state,reward, end_of_video, [quality,rebuf_pen,switch_pen], delay
