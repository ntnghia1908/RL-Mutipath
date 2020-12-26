import numpy as np

class Bola:
    BUFFER_MAX = 30
    def __init__(self):
        self.qualities = np.array([46980,91917,135410,182366,270316,352546,620705]) 
        self.gamma = 5
        self.time = 0
        self.p = 1 # length of each segment
        lw = self.qualities[-1]
        self.v = np.log(self.qualities/lw)
        self.data = {
                "return_buffer": 0,
                "next_seg": [] 
                }
        self.last = 0
        self.f_begin = 0
        self.f_end = 0
    
    def predict(self):
        next_seg = self.data["next_seg"]
        if next_seg == []:
            return 0
        t = self.time
        next_seg = np.flip(next_seg)
        t_p = max(t/2, 3)
        Q = min(t_p,30)
        V = (Q-1)/(self.v[0]+self.gamma)
        m = np.argmax((V*self.v+V*5-self.data["return_buffer"])/next_seg)
        # if m < self.last:
        #     r = self.data["segment"]/self.data["delay"]
        #     temp = next_seg-max(r,lw)
        #     m_p = np.where(temp < 0, temp, -np.inf).argmax()
        #     if m_p<=m:
        #         prediction = m
        #     elif m_p>self.last:
        #         prediction = self.last
        #     else:
        #         prediction = m+1
        # else:
        #     prediction = m
        prediction = m
        # self.last = prediction
        # prediction = len(next_seg)- 1 - prediction 
        return prediction

    def predict2(self):
        next_seg = self.data["next_seg"]
        if next_seg == []:
            return 0
        t = min(self.time,self.f_end)
        tp = max(t/2, 3*self.p)
        Q = min(self.BUFFER_MAX,tp/self.p)
        V = (Q-1)/(self.v[0] + self.gamma*self.p)
        m = np.argmax((V*self.v+V*5-self.data["return_buffer"])/next_seg)
        return m

    def get_data(self, data):
        self.data = data 
        self.time += data["delay"]

    def add_data(self, time, next_seg, delay):
        self.data['return_buffer'] = time[0]
        self.data['next_seg'] = next_seg
        self.time += delay
        self.f_end = time[1]

    def request_buffer(self):
        return None

    def reset(self):
        self.time = 0

