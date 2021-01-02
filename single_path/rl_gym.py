import os
import numpy as np
import rl_agent as rl
from trainning_env import Env

A_DIM = 7
CRITIC_LR_RATE = 0.0001
CHUNK_TIL_VIDEO_END_CAP = 48.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = None
MAX_EP = 10000
EPSILON = 0.1
SAVE_INTERVAL = 500


def main():

    np.random.seed(RANDOM_SEED)
    net_env = Env()
    state = net_env.reset()
    state[-1] = DEFAULT_QUALITY
    agent = rl.policy_gradient_agent(lr=CRITIC_LR_RATE)

    s_batch = []
    a_batch = []
    r_batch = []

    epoch = 0
    score_trace = []
    loss_trace = []
    entropy_record = []
    resume = False

    actor_gradient_batch = []
    critic_gradient_batch = []
    avg_loss = []
    avg_reward = []

    while True:
        if epoch == MAX_EP:
            break
        p = np.random.random_sample()
        if p > EPSILON:
            dist = agent.get_dist(np.array([state]))
            action = agent.pick_action(dist)
        else:
            action = np.random.randint(A_DIM,size=1)[0]
        init_state = state
        state, reward, done, _, _ = net_env.step(action)
        agent.add2batch(init_state, action, reward, done)
        if done:
            state = net_env.reset()
            print(agent.train())
            epoch+=1

    agent.save_model()

if __name__ == '__main__':
    main()