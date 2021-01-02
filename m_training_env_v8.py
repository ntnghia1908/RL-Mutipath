from m_training_env_v7 import PATH1
from numpy.lib.function_base import select
from get_down_size import video_list_collector
import pickle
from numpy import append, zeros, array, vstack, delete
import numpy as np
import math

SAMPLE = 0.02  # sec

# list of events
DOWN = int(1)  # [cur_time, DOWN, down_id, cur_path]
DOWNF = int(2)  # [cur_time, DOWNF, down_id, cur_path]
PLAY = int(3)  # [cur_time, PLAY, play_id, -1]
PLAYF = int(4)  # [cur_time, PLAYF, play_id, -1]
SLEEPF = int(5)  # [cur_time, PLAY, play_id, cur_path]
FREEZEF = int(6)  # [cur_time, PLAY, next_play_id, cur_path]

PATH1 = int(1)
PATH2 = int(2)

 # environment parameters
BITRATE_TRACE1 = 'markovian_bitrate'
BITRATE_TRACE2 = 'markovian_bitrate'
VIDEO_TRACE = 'video_list'
NETWORK_SEGMENT1 = 1  # sec
NETWORK_SEGMENT2 = 1  # sec

HISTORY_SIZE = 6
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 60
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1  # default video quality without agent

BUFFER_THRESH = 32.0  # sec, max buffer limit
MIN_BUFFER_THRESH = 6.0  # sec, min buffer threshold
M_IN_K = 1000.0

RTT1 = 0.080
RTT2 = 0.160

# video bitrate is used as a ultility reward for each bitrate level so this can be change however fit
UTILITY_SCORE = [300, 700, 1200, 1500, 3000, 4000, 5000]  # for 04-second segment
VIDEO_BIT_RATE = [300, 700, 1200, 1500, 3000, 4000, 5000]  # for 04-second segment

VIDEO_CHUNK_LEN = int(4)  # sec, every time add this amount to buffer


class Env():
    SEGMENT_SPACE = 8
    QUALITY_SPACE = 7
    bitrate_list1 = pickle.load(open(BITRATE_TRACE1, 'rb'))
    bitrate_list2 = pickle.load(open(BITRATE_TRACE2, 'rb'))

    def __init__(self):
        super().__init__()
        assert len(UTILITY_SCORE) == len(VIDEO_BIT_RATE)
        # get video list
        vlc = video_list_collector()
        vlc.save_dir = VIDEO_TRACE
        vlc.load()
        self.video_list = vlc.get_trace_matrix(VIDEO_BIT_RATE)

        self.reset()

    def reset(self):
        self.init_net_seg1 = 100
        self.net_seg_id1 = self.init_net_seg1

        self.init_net_seg2 = 150
        self.net_seg_id2 = self.init_net_seg2

        self.network_speed1 = zeros(HISTORY_SIZE)
        self.network_speed2 = zeros(HISTORY_SIZE)

        # initial state, use to lazy initialize the network input
        self.state = append(self.network_speed1, self.network_speed2)
        self.state = append(self.state, zeros(self.QUALITY_SPACE)) ## double check this
        self.state = append(self.state, 0) # Buffer size
        self.state = append(self.state, 0) # last action
        self.state = append(self.state, 0) # remaining chuck

        self.play_id = 0
        self.down_segment = array([-1] * CHUNK_TIL_VIDEO_END_CAP)
        self.down_segment_f = array([-1] * CHUNK_TIL_VIDEO_END_CAP)

        self.reward_qua = 0.0
        self.reward_smooth = 0.0

        self.reward_qua = 0.0
        self.reward_smooth = 0.0
        self.reward_rebuf = 0.0
        self.last_sum_reward = 0.0
        self.est_throughput1 = 0.0
        self.est_throughput2 = 0.0

        self.end_of_video = False
        self.buffer_size = 0
        
        cur_time = 0.0
        self.sleep_time = 0.0


        self.event = [[0.0, DOWN, -1, -1, PATH1]]
        self.down_segment[0] = DEFAULT_QUALITY

        segment_size = float(self.video_list[0][0])
        delay = self.down_time(segment_size, cur_time, PATH1)

        self.event = vstack((self.event, [[delay, DOWNF, int(0), DEFAULT_QUALITY, PATH1]]))
        self.event = vstack((self.event, [[SAMPLE, SLEEPF, -1, -1, PATH2]]))
        self.event = vstack((self.event, [[delay + 1e-4, PLAY, int(0), DEFAULT_QUALITY, -1]]))
        self.event = delete(self.event, 0, 0)
        self.event = self.event[self.event[:, 0].argsort()]
        
        return self.state

    def state_space(self):
        return self.state.shape

    def action_space(self):
        return self.QUALITY_SPACE

    def down_time(self, segment_size, cur_time, path):
        # calculate net_seg_id, seg_time_stamp from cur_time. Remember seg_time_stamp plus rtt
        # set network segment ID to position after sleeping and download last segment
        if (path == PATH1):
            delay = RTT1
            pass_seg = math.floor(cur_time / NETWORK_SEGMENT1)
            self.net_seg_id1 = self.init_net_seg1 + pass_seg
            seg_time_stamp = cur_time - pass_seg

            while True:  # download segment process finish after a full video segment is downloaded
                self.net_seg_id1 = self.net_seg_id1 % len(self.bitrate_list1)  # loop back to begin if finished
                network = self.bitrate_list1[self.net_seg_id1]/2  # network DL_bitrate in bps
                # maximum possible throughput in bytes
                max_throughput = network * (NETWORK_SEGMENT1 - seg_time_stamp)


                if max_throughput > segment_size:  # finish download in network segment
                    seg_time_stamp += segment_size / network  # used time in network segment in second
                    delay += segment_size / network  # delay from begin in second
                    break
                else:
                    delay += NETWORK_SEGMENT1 - seg_time_stamp  # delay from begin in second
                    seg_time_stamp = 0  # used time of next network segment is 0s
                    segment_size -= max_throughput  # remain undownloaded part of video segment
                    self.net_seg_id1 += 1
        if (path == PATH2):
            delay = RTT2
            pass_seg = math.floor(cur_time / NETWORK_SEGMENT2)
            self.net_seg_id2 = self.init_net_seg2 + pass_seg
            seg_time_stamp = cur_time - pass_seg

            while True:
                self.net_seg_id2 = self.net_seg_id2 % len(self.bitrate_list2)  # loop back to begin if finished
                network = self.bitrate_list2[self.net_seg_id2]/2.5  # network DL_bitrate in bps
                max_throughput = network * (NETWORK_SEGMENT2 - seg_time_stamp)  # maximum possible throughput in bytes

                if max_throughput > segment_size:  # finish download in network segment
                    seg_time_stamp += segment_size / network  # used time in network segment in second
                    delay += segment_size / network  # delay from begin in second
                    break
                else:
                    delay += NETWORK_SEGMENT2 - seg_time_stamp  # delay from begin in second
                    seg_time_stamp = 0  # used time of next network segment is 0s
                    segment_size -= max_throughput
                    self.net_seg_id2 += 1
        return delay

    def step(self, action):
        last_down_segment = self.down_segment.copy()
        down_id = self.play_id + math.floor(
            action / self.QUALITY_SPACE) + 1  # self.play_id is playing or just finish playing
        down_quality = action % self.QUALITY_SPACE

        if (down_id > CHUNK_TIL_VIDEO_END_CAP - 1):
            return self.state, -100, self.end_of_video, self.play_id, self.last_sum_reward

        # if down_id has not downloaded yet, but buffer if full then dellete the higher id in buffer
        if ((self.down_segment_f[down_id] < -0.5) and (self.buffer_size > BUFFER_THRESH)):
            id = np.where(self.down_segment_f > -0.5)[-1][-1]
            self.down_segment[id] = -1
            self.down_segment_f[id] = -1
            self.buffer_size -= VIDEO_CHUNK_LEN

        # NEW STEP
        c_even = self.event[0]
        cur_time = self.event[0][0]
        cur_path = self.event[0][4]
        # print(round(cur_time,2), ", DOWN", ", down_id = ", down_id, ", down_quality = ", down_quality, ", buffer_size = ",
        #       self.buffer_size, ", cur_path = ", cur_path)
        self.down_segment[down_id] = down_quality

        segment_size = self.video_list[down_quality][down_id]  # download video segment in bytes
        delay = self.down_time(segment_size, cur_time, cur_path)

        self.event = np.vstack((self.event, [[cur_time + delay, DOWNF, down_id, down_quality, cur_path]]))
        self.event = np.delete(self.event, 0, 0)  # remove the current considering event from event

        self.event = self.event[self.event[:, 0].argsort()]

        if (cur_path == PATH1):
            self.est_throughput1 = segment_size / delay
        if (cur_path == PATH2):
            self.est_throughput2 = segment_size / delay

        while True:
            cur_time = self.event[0][0]
            cur_path = int(self.event[0][4])

            if self.event[0][1] == DOWN:
                break

            if self.event[0][1] == DOWNF:
                # self.buffer_size += VIDEO_CHUNK_LEN
                self.down_segment_f[int(self.event[0][2])] = int(self.event[0][3])
                self.buffer_size = sum((self.down_segment_f[self.play_id:]> -0.5))*VIDEO_CHUNK_LEN
                # print(cur_time, " DOWNF", ", down_id = ", self.event[0][2], ", buffer_size = ", self.buffer_size, ", cur_path = ", cur_path)

                if self.buffer_size > BUFFER_THRESH or (sum(self.down_segment < -0.5) == 0):
                    self.event = np.vstack((self.event, [[cur_time + SAMPLE, SLEEPF, -1, -1, cur_path]]))
                else:
                    self.event = np.vstack((self.event, [[cur_time + 0.0001, DOWN, -1, -1, cur_path]]))


            if self.event[0][1] == SLEEPF:  # this make an infinite loop
                # print(cur_time, " SLEEPF", ", buffer_size = ", self.buffer_size, ", cur_path = ", cur_path)
                self.sleep_time += SAMPLE
                if (self.buffer_size > BUFFER_THRESH) or (sum(self.down_segment < -0.5) == 0):
                    self.event = np.vstack((self.event, [[cur_time + SAMPLE, SLEEPF, -1, -1, cur_path]]))
                else:
                    self.event = np.vstack((self.event, [[cur_time + 0.0001, DOWN, -1, -1, cur_path]]))


            if self.event[0][1] == PLAY:
                self.play_id = int(self.event[0][2])
                self.buffer_size = sum((self.down_segment_f[(self.play_id+1):]>-0.5))*VIDEO_CHUNK_LEN
                # print(cur_time, ", PLAY ", self.play_id, ", buffer_size = ", self.buffer_size)
                # self.buffer_size -= VIDEO_CHUNK_LEN
                play_quality = self.down_segment_f[self.play_id]
                last_play_quality = self.down_segment_f[self.play_id - 1]
                self.reward_qua += UTILITY_SCORE[play_quality]
                self.reward_smooth += np.abs(UTILITY_SCORE[play_quality] - \
                                             UTILITY_SCORE[last_play_quality])
                self.event = np.vstack((self.event, [[cur_time + VIDEO_CHUNK_LEN, PLAYF, self.play_id, play_quality, -1]]))
                # self.event = np.delete(self.event, 0, 0)

            if self.event[0][1] == PLAYF:
                self.play_id = int(self.event[0][2]) # finish play_id

                # print('------------',cur_time, ", PLAYF ", self.play_id, ", buffer_size = ", self.buffer_size)
                # self.play_id = (self.play_id + 1) % len(self.video_list[0]) #loop back if finished video

                # self.play_id += 1
                if self.play_id == CHUNK_TIL_VIDEO_END_CAP - 1:
                    self.event = np.delete(self.event, 0, 0)
                    break

                if self.down_segment_f[self.play_id + 1] < -0.5:
                    self.event = np.vstack((self.event, [[cur_time + SAMPLE, FREEZEF, self.play_id + 1,-1, -1]])) # waiting for play_id+1 chunk
                else:
                    self.event = np.vstack((self.event, [[cur_time + 0.0001, PLAY, self.play_id + 1, self.down_segment_f[self.play_id + 1], -1]]))
                # self.event = np.delete(self.event, 0, 0)

            if self.event[0][1] == FREEZEF:  # this if make an infinite loop
                # print(cur_time, ", FREEZEF", ", buffer_size = ", self.buffer_size)
                # if down_id != 0:  # doesnot add initial delay to rebuf
                self.reward_rebuf += SAMPLE
                if (self.down_segment_f[int(self.event[0][2])] < -0.5):  # next chunk has not downloaded yet
                    self.event = np.vstack((self.event, [[cur_time + SAMPLE, FREEZEF, self.event[0][2], -1, -1]])) # waiting for event[0][2]
                else:
                    self.event = np.vstack((self.event, [[cur_time + 0.0001, PLAY, self.event[0][2], self.down_segment_f[int(self.event[0][3])] , -1]]))

            self.event = np.delete(self.event, 0, 0)  # remove the current considering event from event
            self.event = self.event[self.event[:, 0].argsort()]

        # self.event = np.delete(self.event, 0, 0)  # remove the current considering event from event
        sum_reward = self.reward_qua / M_IN_K - SMOOTH_PENALTY * \
                     self.reward_smooth / M_IN_K - self.reward_rebuf * REBUF_PENALTY
        reward = sum_reward - self.last_sum_reward
        self.last_sum_reward = sum_reward

        # CALCULATE NEW STATE
        next_chunk_size = np.array([])
        chunk_state = np.array([])
        for i in range(self.play_id, self.play_id + self.SEGMENT_SPACE):
            if i < CHUNK_TIL_VIDEO_END_CAP:
                if (self.down_segment[i] < -0.5):
                    chunk_state = np.append(chunk_state, 0)
                else:
                    chunk_state = np.append(chunk_state, 1)
                next_chunk_size = np.append(next_chunk_size, self.video_list[:, i])
            else:
                next_chunk_size = np.append(next_chunk_size, np.array([0] * self.QUALITY_SPACE))
                chunk_state = np.append(chunk_state, 1)

        self.network_speed1 = np.roll(self.network_speed1, axis=-1, shift=1)
        self.network_speed1[0] = self.est_throughput1 / 1e6

        self.network_speed2 = np.roll(self.network_speed2, axis=-1, shift=1)
        self.network_speed2[0] = self.est_throughput2 / 1e6

        remain = CHUNK_TIL_VIDEO_END_CAP - self.play_id

        self.state = np.append(self.network_speed1, self.network_speed2)  # est throughput1, est throughput 2
        self.state = np.append(self.state, next_chunk_size / M_IN_K / M_IN_K)  # next video chunk
        self.state = np.append(self.state, chunk_state)  # which chunk is downloaded
        self.state = np.append(self.state, self.buffer_size / BUFFER_NORM_FACTOR)  # buffer size
        # self.state = np.append(self.state, remain)  # remain video chunks
        self.state = np.append(self.state, self.down_segment[self.play_id - 1] * 0.1)  # last quality


        if (last_down_segment[down_id] > -0.5 ):
            reward = -100

        if self.play_id == CHUNK_TIL_VIDEO_END_CAP - 1:  # if terminate reset
            self.end_of_video = True

        return self.state, reward, self.end_of_video, self.play_id, sum_reward

def test():
    env = Env()
    env.reset()

if __name__ == "__main__":
    test()