import matplotlib.pyplot as plt
import pickle
import numpy as np
import rl_agent
from trainning_env import Env

# SGD.pickle1588175032275
# SGD.pickle1588176589443

# This is evaluation for policy_gradient_agent require model/SGDmodel.h5 to be generated
env = Env()
agent = rl_agent.policy_gradient_agent()
state = env.reset()
agent.get_dist(np.array([state]))
agent.load_model()
data = pickle.load(open('agent/SGD.pickle','rb'))
in_data = data['score_trace']
plot_data = []
norm = 400
for i in range(len(in_data)-norm):
    plot_data.append(np.average(in_data[i:i+norm]))
plt.plot(plot_data, label = "Policy gradient", color = "blue")

# This is evaluation for actor_critic_agent require model/A2Cmodel.h5 to be generated
# env = Env()
# agent = rl_agent.actor_critic_agent()
# state = env.reset()
# agent.get_dist(np.array([state]))
# agent.load_model()
# data = pickle.load(open('agent/A2C.pickle','rb'))
# in_data = data['score_trace']
# plot_data = []
# for i in range(len(in_data)-norm):
#     plot_data.append(np.average(in_data[i:i+norm]))
# plt.plot(plot_data, label = "A2C", color = "red")
plt.title("Average reward of {} consecutive episode".format(norm))

plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()
