import numpy as np

class SmoothThroughput:
    HISTORY_STORY = 6
    SAFE_THRESHOLD = 0.90

    def __init__(self):

        self.action_list = [46980,91917,135410,182366,270316,352546,620705]
        self.history = np.zeros(self.HISTORY_STORY)

    def predict(self, next_seg):
        predict =  np.mean(self.history)
        picked_quality = 1
        while(next_seg[picked_quality]<predict ):
            picked_quality += 1
            if picked_quality == self.action_list.__len__():
                break
        return picked_quality - 1

    def add(self,througput):
        self.history = np.roll(self.history,1)
        self.history[0] = througput
