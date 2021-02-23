from m_training_env_v7 import Env
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
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations

        observations, reward, done, _, _ = env.step(action)

        rewards += reward

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

    env.reset()
    return rewards, mean(losses)



def main():
    pretrain = True
    env = Env()
    gamma = 0.9
    copy_step = 25
    num_states = env.state_space()
    num_actions = env.action_space()
    hidden_units = [128, 256]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 1000
    lr = 1e-3
    file_name = 'multi_path_decay99995'
    current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    N = 50000
    total_rewards = np.array([])
    epsilon = 0.99
    decay = 0.99995
    min_epsilon = 0.1
    epoch = []
    avg_rewards = []

    for n in range(N):
        epoch.append(n)
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_video(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards = np.append(total_rewards, [total_reward])
        avg_reward = total_rewards[max(0, n - 100):(n + 1)].mean()
        avg_rewards.append(avg_reward)

        print('epoch:{} reward:{}'.format(n, total_rewards[-1]))
        if n != 0 and n % 500 == 0:
            plt.figure(figsize=(15,3))
            plt.plot(epoch, total_rewards)
            plt.plot(epoch, avg_rewards)
            plt.savefig('fig/{}.png'.format(file_name))
            plt.clf()
            # f = 'downtrack/downtrack{}.csv'.format(n)
            f2 = 'rewards/history_{}'.format(file_name)
            np.save(f2, [total_rewards, avg_rewards, epsilon])
            if n % 5000 == 0:
                TrainNet.save_model("model/DQNmodel_{}.h5".format(file_name))


if __name__ == '__main__':
    main()
