import numpy as np
import math
from .get_down_size import video_list_collector


VIDEO_CHUNK_LEN = 4                            # sec, every time add this amount to buffer
QUALITY_LEVELS = 7
TOTAL_VIDEO_CHUNK = 60                          # number of video segment per training section
BUFFER_THRESH = 30.0                            # sec, max buffer limit
MIN_BUFFER_THRESH = 6            # sec, min buffer threshold
SAMPLE = 0.2                   # sec
NETWORK_SEGMENT = 1                             # sec
rtt = 0.080 

# events
DOWN = int(2)
DOWNF = int(1)
PLAY = int(4)
PLAYF = int(3)
SLEEPF = int(5)
FREEZEF = int(6)

class VideoPlayer():
    def __init__(self, bitrate_list, video_list):

        self.bitrate_list = bitrate_list 
        self.video_list = video_list

        self.buffer_thresh = BUFFER_THRESH
        self.reset()

        
    def reset(self, mode = 'train'):
        if mode == 'test':
            self.net_seg_id = 0 # current network segment in the bitrate_list
            self.down_id = 0 # similar to Quan's video_id, the video segment ID currently downloading
            self.play_id = 0 # the video segment ID currently playing

        else:
            # self.net_seg_id = 0 # current network segment in the bitrate_list
            # self.down_id = 0 # similar to Quan's video_id, the video segment ID currently downloading

            self.init_net_seg = np.random.randint(0, len(self.bitrate_list)-1)
            self.net_seg_id =  10
            # self.down_id = np.random.randint(0, len(self.video_list[0])-1)
            self.down_id = 0
            self.play_id = self.down_id  

        self.video_seg_download = 0 # the number of video segments have been downloaded. If self.video_seg_download >= TOTAL_VIDEO_CHUNK then terminate, reset and start a new episode

        self.buffer = 0
        self.event = [[0, DOWN, self.down_id]]
        # print("Download segment: ", self.down_id, ", Buffer = ", self.buffer)

        self.terminate = False

        # download the first segment at the lowest quality_level
        segment_size = float(self.video_list[0][self.down_id])   #download video segment in bytes                       
        delay = self.down_time(segment_size, 0)
        # add DOWNF and FREEZEF events to event[]
        self.event = np.vstack((self.event,[[delay, DOWNF, self.down_id]]))
        self.event = np.vstack((self.event,[[delay+0.001, FREEZEF, 0]]))
        self.event = np.delete(self.event,0,0) # remove the current considering event from event

        return self.video_list[:,self.down_id] 


    def down_time(self,segment_size, cur_time):
        # calculate net_seg_id, seg_time_stamp from cur_time. Remember seg_time_stamp plus rtt
        # set network segment ID to position after sleeping and download last segment
        delay = rtt
        pass_seg = math.floor(cur_time / NETWORK_SEGMENT)
        self.net_seg_id = self.init_net_seg + pass_seg
        seg_time_stamp = cur_time - pass_seg

        while True:                                                                 #download segment process finish after a full video segment is downloaded
            self.net_seg_id = self.net_seg_id % len(self.bitrate_list)  #loop back to begin if finished
            network = self.bitrate_list[self.net_seg_id]                      #network DL_bitrate in bps
            max_throughput = network * (NETWORK_SEGMENT - seg_time_stamp)      #maximum possible throughput in bytes

            if max_throughput > segment_size:                                        #finish download in network segment
                seg_time_stamp += segment_size / network                        #used time in network segment in second
                delay += segment_size / network                                      #delay from begin in second
                break
            else:                                                               
                delay += NETWORK_SEGMENT - seg_time_stamp                  #delay from begin in second
                seg_time_stamp = 0                                         #used time of next network segment is 0s
                segment_size -= max_throughput                                       #remain undownloaded part of video segment
                self.net_seg_id +=1
        return delay

    def download(self, quality_level):
        # This function is called when downloading a new segment 
        # events is the axis of events start from the download event. 
        # first column is timestamp, 2nd column is type of event
        assert quality_level >=0
        assert quality_level < QUALITY_LEVELS
        # print("Download segment: ", self.down_id, ", Buffer = ", self.buffer)

        cur_time = self.event[0][0] 
        finish = self.terminate    
        segment_size = float(self.video_list[quality_level][self.down_id])   #download video segment in bytes               
        delay = self.down_time(segment_size, cur_time)
        self.event = np.vstack((self.event,[[cur_time+delay, DOWNF, self.down_id]]))
        self.event = np.delete(self.event,0,0) # remove the current considering event from event
        rebuf = 0
        sleep_time = 0
        while True:
            cur_time = self.event[0][0]
            # print("Current time: ", cur_time, ", Event: ", self.event[0][1], ", Buffer = ", self.buffer)
            if self.event[0][1] == DOWN:
                break

            if self.event[0][1] == DOWNF:
                self.buffer += VIDEO_CHUNK_LEN
                # print(self.event[0][0], " Finish down segment ", self.down_id, ", Buffer = ", self.buffer)                
                #self.down_id = (self.down_id + 1) % len(self.video_list[0]) #loop back if finished video
                self.down_id += 1
                self.video_seg_download += 1

                if self.buffer > self.buffer_thresh: 
                   self.event = np.vstack((self.event,[[cur_time + SAMPLE, SLEEPF,0]]))
                else:
                    self.event = np.vstack((self.event,[[cur_time+0.0001, DOWN,self.down_id]]))
                self.event = np.delete(self.event,0,0) # remove the current considering event from event

            if self.event[0][1] == SLEEPF:
                sleep_time += SAMPLE
                # print(self.event[0][0], " SLEEPF", ", Buffer = ", self.buffer)
                if self.buffer > self.buffer_thresh: 
                    self.event = np.vstack((self.event,[[cur_time + SAMPLE, SLEEPF,0]]))
                else:
                    self.event = np.vstack((self.event,[[cur_time+0.0001, DOWN,self.down_id]]))
                self.event = np.delete(self.event,0,0) # remove the current considering event from event

            if self.event[0][1] == PLAY:
                self.buffer -= VIDEO_CHUNK_LEN           
                # print(self.event[0][0], " Play segment ", self.play_id, ", Buffer = ", self.buffer)  
                self.event = np.vstack((self.event,[[cur_time + VIDEO_CHUNK_LEN, PLAYF,self.play_id]]))
                self.event = np.delete(self.event,0,0) # remove the current considering event from event

            if self.event[0][1] == PLAYF:
                # print(self.event[0][0], " Finish play segment ", self.play_id, ", Buffer = ", self.buffer)                
                # self.play_id = (self.play_id + 1) % len(self.video_list[0]) #loop back if finished video
                self.play_id += 1 
                if self.buffer < MIN_BUFFER_THRESH:
                    self.event = np.vstack((self.event,[[cur_time + SAMPLE, FREEZEF,0]]))
                else:
                    self.event = np.vstack((self.event,[[cur_time+0.0001, PLAY, self.play_id]]))
                self.event = np.delete(self.event,0,0) # remove the current considering event from event

            if self.event[0][1] == FREEZEF:
                if self.down_id != 0:
                    rebuf += SAMPLE
                #print(self.event[0][0], " FREEZEF", ", Buffer = ", self.buffer)  
                if self.buffer < MIN_BUFFER_THRESH:
                    self.event = np.vstack((self.event,[[cur_time + SAMPLE, FREEZEF,0]]))
                else:
                    self.event = np.vstack((self.event,[[cur_time+0.0001, PLAY,self.play_id]]))
                self.event = np.delete(self.event,0,0) # remove the current considering event from event

            self.event = self.event[self.event[:,0].argsort()]
            # print(self.event)
        if self.down_id >= TOTAL_VIDEO_CHUNK:                       #if terminate reset
            finish = True
            self.reset()
        
        next_segments = self.video_list[:,self.down_id]                 #get sizes of next download options
        remain = TOTAL_VIDEO_CHUNK - self.video_seg_download                
        # print("Rebuf = ", rebuf, ", Sleeptime = ", sleep_time, ", delay = ", delay, ", remain = ", remain)
        return delay, sleep_time, self.buffer, rebuf, segment_size, next_segments, finish, remain , self.bitrate_list[self.net_seg_id]

    def set_buffer(self, size):
        if size != None:
            self.buffer_thresh = size

    def get_buffer_thresh(self):
        return self.buffer_thresh