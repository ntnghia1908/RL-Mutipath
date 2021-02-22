import numpy as np
from m_training_env_v11 import Env
import mSmoothThroughput as smooth

EVAL_EPS = 100
env = Env()
state = env.reset()
ep = 0
throughput_rewards = []

while True:
    if ep == EVAL_EPS:
        break

    action = smooth.predict(state, cur_path) # next segment list
    state, reward, done, sep_reward, _ = net_env.step(action)
    throughput.add(state[0] * 1e6) # add network speed
    total_reward += reward
    if done:
        ep += 1
        throughput_rewards.append(total_reward)
        total_reward = 0
        state = net_env.reset()

# convert to cdf
throughput_rewards = np.array(throughput_rewards)
throughput_rewards = np.sort(throughput_rewards)
y_axis = np.arange(1,len(throughput_rewards)+1) / len(throughput_rewards)
plt.plot(throughput_rewards,y_axis,label = 'throughput',color="red")