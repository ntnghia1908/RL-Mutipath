## In version 14 of environment, we implement pretrain and real trace

from m_training_env_v14 import Env
from statistics import mean
from cart_pole import DQN
from helper import write2csv, np2csv
import matplotlib.pyplot as plt

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import datetime


reward_trace = [['reward_quality', 'smooth_penalty', 'rebuf_penalty', 'sum_reward']]

def play_video(env, TrainNet, TargetNet, epsilon, copy_step):
    total_reward = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    history = ""
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations


        observations, reward, done, line, _ , _ , _, _  = env.step(action)

        history += line
        total_reward += reward

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

    write2csv('play_event', env.play_event)
    write2csv('down_event', env.down_event)
    reward_trace.append(env.reward_trace)

    env.reset()
    with open('history.txt', 'w+') as f:
        f.write(history)
    return total_reward, mean(losses)


def main():
    env = Env()
    gamma = 0.9
    copy_step = 25
    num_states = env.state_space()
    num_actions = env.action_space()
    hidden_units = [128, 256]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 200
    lr = 1e-4*0.5

    N = 1000
    total_rewards = np.array([])
    epsilon = 0.99
    decay = 0.9
    min_epsilon = 0.1
    epoch = []
    avg_rewards = []
    file_name = 'lr45_decay9_batch200_fixtrace_v14'
    current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    state = env.reset()

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    PRETRAIN = False
    if PRETRAIN == True:
        epsilon = 0.1
        decay = 0
        min_epsilon = 0.1
        TrainNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_lr45_decay999_batch200_v14.h5')
        TargetNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_lr45_decay999_batch200_v14.h5')

    TEST = False
    if TEST == True:
        epsilon = 0.0
        decay = 1.0
        min_epsilon = 0.0
        N = 100
        file_name = file_name+'_test'
        TrainNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_lr45_decay999_batch200_v14.h5')
        TargetNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_lr45_decay999_batch200_v14.h5')


    for n in range(N):
        epoch.append(n)
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_video(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards = np.append(total_rewards, [total_reward])
        avg_reward = total_rewards[max(0, n - 100):(n + 1)].mean()
        avg_rewards.append(avg_reward)

        # print('epoch:{} reward:{}'.format(n, total_rewards[-1]))
        # if n != 0 and n % 10 == 0:
        #     plt.figure(figsize=(15,3))
        #     np2csv('total_rewards', total_rewards)
        #     write2csv('reward_trace', reward_trace)
        #     plt.plot(epoch, total_rewards)
        #     plt.plot(epoch, avg_rewards)
        #     plt.savefig('fig/{}.png'.format(file_name))
        #     plt.clf()
        #     # f = 'downtrack/downtrack{}.csv'.format(n)
        #     f2 = 'rewards/history_{}'.format(file_name)
        #     np.save(f2, [total_rewards, avg_rewards, epsilon])
        #     # if n % 10 == 0:
        #     TrainNet.save_model("model/DQNmodel_{}.h5".format(file_name))

        print('epoch:{} reward:{}'.format(n, total_rewards[-1]))
        if n != 0 and n % 50 == 0:
            plt.figure(figsize=(15,3))
            plt.plot(epoch, total_rewards)
            plt.plot(epoch, avg_rewards)
            plt.savefig('fig/{}.png'.format(file_name))
            plt.clf()
            # f = 'downtrack/downtrack{}.csv'.format(n)
            f2 = 'rewards/history_{}'.format(file_name)
            np.save(f2, [total_rewards, avg_rewards, epsilon])
            if n % 50 == 0:
                TrainNet.save_model("model/DQNmodel_{}.h5".format(file_name))


if __name__ == '__main__':
    main()
