from simulator.m_training_env_v6 import Env
from statistics import mean
from cart_pole import DQN
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime
import tensorflow as tf

def play_video(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    downtracks = []
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, play_id, downtrack = env.step(action)
        downtracks.append(downtrack)
        
        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp = {'s': prev_observations,
               'a': action,
               'r': reward,
               's2': observations,
               'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TrainNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses), downtracks



def main():
    env = Env()
    gamma = 0.99
    copy_step = 25
    num_states = 78
    num_actions = 56
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 50000
    total_rewards = np.array([])
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    epoch = []
    avg_rewards = []

    for n in range(N):
        epoch.append(n)
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses, downtracks = play_video(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards = np.append(total_rewards, [total_reward])
        avg_reward = total_rewards[max(0, n - 100):(n + 1)].mean()
        avg_rewards.append(avg_reward)

        print('epoch:{} reward:{}'.format(n, total_rewards[-1]))
        if n!=0 and n % 100 == 0:
            plt.figure(figsize=(15,3))
            plt.plot(epoch, total_rewards)
            plt.plot(epoch, avg_rewards)
            plt.savefig('figure/episode{}.png'.format(n))
            f = 'downtrack/downtrack{}.csv'.format(n)
            pd.DataFrame(downtracks).to_csv(f)
        
if __name__ == '__main__':
    main()
    # print("hello")
    # draw()
