from trainning_env import Env
from policy_agent import Agent
import tensorflow as tf
import numpy as np

env = Env()

gamma = 0.99
def discount_rewards(r):
    
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

h_size, t_size = env.state_space()
a_size = env.action_space()

myAgent = Agent(h_size,t_size,a_size,lr=1e-2)
action_space = env.action_list
total_episodes = 5000
update_frequency = 50

i = 0
e = 0.1
total_reward = []
total_length = []
gradBuffer = myAgent.get_weights()

for ix,grad in enumerate(gradBuffer):
    gradBuffer[ix] =grad*0

while i < total_episodes:
    s = env.reset()
    running_reward = 0
    ep_history = []
    a = 0 
    j = 0
    while True:
        j += 1
        a = myAgent.choose_action(np.array([s]))
        if np.random.uniform(0,1) < e :
            a = np.random.randint(0,action_space)
        s1,r,d,_ = env.step(a)
        ep_history.append([s,a,r,s1])
        s = s1
        running_reward += r
        if d == True:

            ep_history = np.array(ep_history)
            ep_history[:,2] = discount_rewards(ep_history[:,2])
            with tf.GradientTape() as tape:
                loss = myAgent.loss(np.float32(ep_history[:,2]), np.int32(ep_history[:,1]), np.vstack(ep_history[:,0]))
                tvar = myAgent.get_weights()
            grads = tape.gradient(loss,tvar)
            for idx,grad in enumerate(grads):
                gradBuffer[idx] += grad

            if i % update_frequency == 0 and i!=0:
                myAgent.update_batch(gradBuffer)
            
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0
            
            total_reward.append(running_reward)
            total_length.append(j)
            break
    if i % 100 == 0:
        print(str(np.mean(total_reward[-100:])) + ' ' + str(i))
    i += 1

myAgent.save_model()