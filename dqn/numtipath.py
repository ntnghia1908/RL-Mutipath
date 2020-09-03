from simulator.m_training_env_v6 import Env
from statistics import mean
from cart_pole import DQN
import matplotlib.pyplot as ptl

import numpy as np
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
        observations, reward, done, play_id, sum_reward = env.step(action)
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
    return rewards, mean(losses)



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
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 50000
    total_rewards = []
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    epoces = []

    for n in range(N):
        epoces.append(n)
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_video(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards.append(total_reward)
        # avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()

        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            # tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss)', losses, step=n)
        print('epoces:{} reward:{}'.format(n, total_rewards))
        if n % 100 == 0:
            ptl.plot(epoces, total_rewards)
            ptl.savefig('episode{}.png'.format(n))
            # print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
            #       "episode loss: ", losses)
    # print("avg reward for last 100 episodes:", avg_rewards)


if __name__ == '__main__':
    main()
