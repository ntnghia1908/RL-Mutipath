import rl_agent as rl
import SmoothThroughput as smooth
import bola_basic as bola
import numpy as np
from trainning_env import Env
import matplotlib.pyplot as plt

EVAL_EPS = 1000
DEFAULT_QUALITY = 1

net_env = Env()
state = net_env.reset()

# evaluation for rl agent
agent = rl.policy_gradient_agent()
agent.get_dist(np.array([state]))
agent.load_model()

ep = 0

utility_reward = 0
rebuf_penalty  = 0
smooth_penalty = 0

agent_utility = []
agent_rebuf = []
agent_smooth = []

utility_list = []
rebuf_list = []
smooth_list = []


while True:
    if ep == EVAL_EPS:
        break

    dist = agent.get_dist(np.array([state]))
    action = agent.pick_action(dist)
    state, reward, done, sep_reward, _ = net_env.step(action)
    utility_reward += sep_reward[0]
    rebuf_penalty += sep_reward[1]
    smooth_penalty += sep_reward[2]

    if done:
        ep += 1
        agent_utility.append(utility_reward)
        agent_rebuf.append(rebuf_penalty)
        agent_smooth.append(smooth_penalty)

        utility_reward = 0
        rebuf_penalty  = 0
        smooth_penalty = 0
        state = net_env.reset()

agent_utility = np.asarray(agent_utility)
agent_utility = np.average(agent_utility)
utility_list.append(agent_utility)
agent_smooth = np.asarray(agent_smooth)
agent_smooth = np.average(agent_smooth)
smooth_list.append(agent_smooth)
agent_rebuf = np.asarray(agent_rebuf)
agent_rebuf = np.average(agent_rebuf)
rebuf_list.append(agent_rebuf)

# evaluation for rl agent
agent = rl.actor_critic_agent()
agent.get_dist(np.array([state]))
agent.load_model()

ep = 0

utility_reward = 0
rebuf_penalty  = 0
smooth_penalty = 0

agent_utility = []
agent_rebuf = []
agent_smooth = []

while True:
    if ep == EVAL_EPS:
        break

    dist = agent.get_dist(np.array([state]))
    action = agent.pick_action(dist)
    state, reward, done, sep_reward, _ = net_env.step(action)
    utility_reward += sep_reward[0]
    rebuf_penalty += sep_reward[1]
    smooth_penalty += sep_reward[2]
    if done:
        ep += 1
        
        agent_utility.append(utility_reward)
        agent_rebuf.append(rebuf_penalty)
        agent_smooth.append(smooth_penalty)
        utility_reward = 0
        rebuf_penalty  = 0
        smooth_penalty = 0
        state = net_env.reset()

agent_utility = np.asarray(agent_utility)
agent_utility = np.average(agent_utility)
utility_list.append(agent_utility)
agent_smooth = np.asarray(agent_smooth)
agent_smooth = np.average(agent_smooth)
smooth_list.append(agent_smooth)
agent_rebuf = np.asarray(agent_rebuf)
agent_rebuf = np.average(agent_rebuf)
rebuf_list.append(agent_rebuf)

# evaluation for smooth throughput
state = net_env.reset()
ep = 0

utility_reward = 0
rebuf_penalty  = 0
smooth_penalty = 0

agent_utility = []
agent_rebuf = []
agent_smooth = []

throughput = smooth.SmoothThroughput()

while True:
    if ep == EVAL_EPS:
        break

    action = throughput.predict(state[9:-1]*1e6) # next segment list
    state, reward, done, sep_reward, _ = net_env.step(action) 
    throughput.add(state[0] * 1e6) # add network speed
    utility_reward += sep_reward[0]
    rebuf_penalty += sep_reward[1]
    smooth_penalty += sep_reward[2]
    if done:
        ep += 1
        agent_utility.append(utility_reward)
        agent_rebuf.append(rebuf_penalty)
        agent_smooth.append(smooth_penalty)
        utility_reward = 0
        rebuf_penalty  = 0
        smooth_penalty = 0
        state = net_env.reset()

agent_utility = np.asarray(agent_utility)
agent_utility = np.average(agent_utility)
utility_list.append(agent_utility)
agent_smooth = np.asarray(agent_smooth)
agent_smooth = np.average(agent_smooth)
smooth_list.append(agent_smooth)
agent_rebuf = np.asarray(agent_rebuf)
agent_rebuf = np.average(agent_rebuf)
rebuf_list.append(agent_rebuf)

# evaluation for bola basic
state = net_env.reset()
ep = 0

utility_reward = 0
rebuf_penalty  = 0
smooth_penalty = 0

agent_utility = []
agent_rebuf = []
agent_smooth = []

bola_agent = bola.Bola()
bola_agent.add_data(state[7:9]*10,state[9:-1]*1e6,0)

while True:
    if ep == EVAL_EPS:
        break

    action = bola_agent.predict2()
    state, reward, done, sep_reward, delay = net_env.step(action) 
    bola_agent.add_data(state[7:9]*10,state[9:-1]*1e6,delay)
    utility_reward += sep_reward[0]
    rebuf_penalty += sep_reward[1]
    smooth_penalty += sep_reward[2]
    if done:
        ep += 1
        agent_utility.append(utility_reward)
        agent_rebuf.append(rebuf_penalty)
        agent_smooth.append(smooth_penalty)
        utility_reward = 0
        rebuf_penalty  = 0
        smooth_penalty = 0
        state = net_env.reset()

agent_utility = np.asarray(agent_utility)
agent_utility = np.average(agent_utility)
utility_list.append(agent_utility)
agent_smooth = np.asarray(agent_smooth)
agent_smooth = np.average(agent_smooth)
smooth_list.append(agent_smooth)
agent_rebuf = np.asarray(agent_rebuf)
agent_rebuf = np.average(agent_rebuf)
rebuf_list.append(agent_rebuf)

print(utility_list)
fig, ax = plt.subplots()
ind = np.arange(4)
width = 0.2

p1 = ax.bar(ind, utility_list, width, bottom = 0)
p2 = ax.bar(ind+width, smooth_list, width, bottom = 0)
p3 = ax.bar(ind+2*width, rebuf_list, width, bottom = 0)
ax.set_title('Scores agents and categories')
ax.set_xticks(ind + width)
ax.set_xticklabels(('SGD','A2C','Smooth','BOLA'))

plt.ylabel("total score")
ax.legend((p1[0], p2[0], p3[0]), ('utility score','smooth penalty','rebuf penalty'))
plt.show()

# plt.title("Performance CDF")

# plt.legend()
# plt.show()
