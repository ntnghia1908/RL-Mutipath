import pickle
import numpy as np
import copy
from .video_player import VideoPlayer
from .get_down_size import video_list_collector


class Env():
    BITRATE_TRACE = 'simulator/traces/markovian_bitrate'
    VIDEO_TRACE = 'simulator/video/video_list'
    A_DIM = 7
    HISTORY_SIZE = 7  
    TRACE_SIZE = 8  # take how many frames in the past
    BUFFER_NORM_FACTOR = 10.0
    CHUNK_TIL_VIDEO_END_CAP = 60.0
    M_IN_K = 1000.0
    REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
    SMOOTH_PENALTY = 1
    DEFAULT_QUALITY = 1  # default video quality without agent
    # video bitrate is used as a ultility reward for each bitrate level so this can be change however fit
    UTILITY_SCORE = [60,170,340,1010,2080,4130,5900]
    VIDEO_BIT_RATE = [46980,135410,270316,808057,1662809,3305118,4726737] # unit is bit
    
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

    def state_space(self):
        return self.HISTORY_SIZE, self.TRACE_SIZE

    def action_space(self):
        return self.action_list

    def reset(self, mode='train'):
        next_video_chunk_size = self.video_player.reset(mode = mode)
        self.last_action = self.DEFAULT_QUALITY
        self.network_speed = np.zeros(self.HISTORY_SIZE)
        self.state = np.append(self.network_speed, 0)
        self.state = np.append(self.state, next_video_chunk_size)
        self.state = np.append(self.state, 0)

        terminate = False
        reward = 0
        return self.state

    def process_trace(self, data):
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

    def step_new(self,action,format = 0):
        last_state = self.state
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            self.video_player.download(action)


        self.state = np.roll(self.state, -1, axis=1)

        self.state[0, -1] = self.UTILITY_SCORE[action] / float(np.max(self.UTILITY_SCORE)) * 10 
        self.state[1, -1] = buffer_size / self.BUFFER_NORM_FACTOR  *10
        self.state[2, -1] = float(video_chunk_size) / float(delay) / self.M_IN_K /10 
        self.state[3, -1] = float(delay)
        self.state[4, :self.A_DIM] = np.array(next_video_chunk_sizes) / self.M_IN_K /100
        self.state[5, -1] = np.minimum(video_chunk_remain, self.CHUNK_TIL_VIDEO_END_CAP) / float(self.CHUNK_TIL_VIDEO_END_CAP) *10
        reward = self.cal_reward(action, rebuf)
        self.last_action = action
        if format == 1:
            return video_chunk_size / delay, delay, np.array(next_video_chunk_sizes), reward, end_of_video
        return self.state, reward, end_of_video, False, delay

    def step(self,action):
        last_state = self.state
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            self.video_player.download(action)

        reward = self.cal_reward(action, rebuf)
        self.last_action = action
        net_speed = video_chunk_size / (delay - sleep_time)
        self.network_speed = np.roll(self.network_speed,axis = -1,shift = 1)
        self.network_speed[0] = net_speed
        self.state = np.append(self.network_speed, buffer_size * 1e3)
        self.state = np.append(self.state, next_video_chunk_sizes)
        self.state = np.append(self.state, self.last_action * 1e4)
        
        return self.state,reward, end_of_video, False, delay 