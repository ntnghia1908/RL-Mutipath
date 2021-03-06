import numpy as np
from helper import write2csv
from m_training_env_v15_test import Env
import matplotlib.pyplot as plt
import helper
from get_down_size import video_list_collector

M_IN_K = 1000.0

HISTORY_SIZE = 6
SEGMENT_SPACE = 8
QUALITY_SPACE = 7
CHUNK_TIL_VIDEO_END_CAP = 60

SCALE = 0.9

# get video list
VIDEO_TRACE = 'video_list'
VIDEO_BIT_RATE = [300, 700, 1200, 1500, 3000, 4000, 5000]  # for 04-second segment
VIDEO_CHUNK_LEN = int(4)  # sec, every time add this amount to buffer

vlc = video_list_collector()
vlc.save_dir = VIDEO_TRACE
vlc.load()
video_list = vlc.get_trace_matrix(VIDEO_BIT_RATE)

def get_downId(down_segment):
    not_down_yet = np.where(down_segment < -0.5)[0]
    return not_down_yet[0]   # some problem here


def predict(env, state, down_id, cur_path, prev_segment=3):
    segment_bitrates = video_list[:,down_id] * 8.0 / VIDEO_CHUNK_LEN  # average video bitrates of down_id segment in bits per second
    history_net_speed = None

    if cur_path == 1:
        history_net_speed = state[:HISTORY_SIZE]
    elif cur_path == 2:
        history_net_speed = state[HISTORY_SIZE:2*HISTORY_SIZE]
    else:
        raise AttributeError("cur_path not exit")

    predict = np.mean(history_net_speed[:prev_segment]) # calculated in Mbps

    picked_quality = 0
    if segment_bitrates[picked_quality] >= predict * SCALE * M_IN_K * M_IN_K:
        return picked_quality

    while segment_bitrates[picked_quality] < predict * SCALE * M_IN_K * M_IN_K:
        picked_quality += 1
        if picked_quality == env.QUALITY_SPACE:
            break

    return picked_quality - 1


def evaluate(env):
    EVAL_EPS = 150
    state = env.reset()
    throughput_rewards = []
    total_reward = 0.0
    down_segment = np.array([-1] * CHUNK_TIL_VIDEO_END_CAP)
    play_id = 0

    ep = 1
    reward_trace = [['reward_quality', 'smooth_penalty', 'rebuf_penalty', 'sum_reward']]
    while True:
        if ep > EVAL_EPS:
            break
        cur_path = 1
        done = False

        while not done:
            # chunk_state = state[HISTORY_SIZE*2 + QUALITY_SPACE * SEGMENT_SPACE:HISTORY_SIZE*2 + QUALITY_SPACE * SEGMENT_SPACE+SEGMENT_SPACE]
            # down_id = get_downId(chunk_state)
            down_id = get_downId(down_segment)
            quality = predict(env, state, down_id, cur_path)
            action = (down_id - play_id -1) * QUALITY_SPACE + quality
            # print("play_id, down_id, quality, cur_path", play_id, down_id, quality, cur_path)

            state, reward, done, sep_reward, play_id, _, cur_path, down_segment = env.step(action)
            total_reward += reward

            if done:
                ep += 1
                throughput_rewards.append(total_reward)

                write2csv('evaluateSmooth/{}play_event'.format(round(total_reward, 1)), env.play_event)
                write2csv('evaluateSmooth/{}down_event'.format(round(total_reward, 1)), env.down_event)
                write2csv('evaluateSmooth/{}buffer_traces'.format(round(total_reward, 1)), env.buffer_size_trace)
                write2csv('evaluateSmooth/{}bw1'.format(round(total_reward, 1)), env.bw1_trace)
                write2csv('evaluateSmooth/{}bw2'.format(round(total_reward, 1)), env.bw2_trace)

                quality_reward = env.var1
                smooth_reward = env.var2
                buffering_reward = env.var3

                reward_trace.append([quality_reward, smooth_reward, buffering_reward, total_reward])
                print(ep, quality_reward, smooth_reward, buffering_reward, total_reward)

                state = env.reset()
                # throughput_rewards = []
                total_reward = 0.0
                down_segment = np.array([-1] * CHUNK_TIL_VIDEO_END_CAP)
                play_id = 0

        write2csv('evaluateSmooth/reward_trace', reward_trace)


    # # convert to cdf
    # throughput_rewards = np.array(throughput_rewards)
    # throughput_rewards = np.sort(throughput_rewards)
    # helper.np2csv('throughput_retrace_reward', throughput_rewards)
    # y_axis = np.arange(1, len(throughput_rewards) + 1) / len(throughput_rewards)
    # plt.plot(throughput_rewards, y_axis, label='throughput', color="red")
    # plt.show()


if __name__ == '__main__':
    print("seek", 13)
    np.random.seed(13)
    env = Env()
    evaluate(env)
