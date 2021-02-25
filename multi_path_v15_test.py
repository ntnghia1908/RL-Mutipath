## In version 14 of environment, we implement pretrain and real trace

from m_training_env_v15_test import Env
from statistics import mean
from cart_pole import DQN
from helper import write2csv, np2csv, plot_reward
import matplotlib.pyplot as plt

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import datetime

def play_video(env, TrainNet, TargetNet, epsilon, copy_step):
    total_reward = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    history = ""
    while not done:
        quality_reward = 0.0
        smooth_reward = 0.0
        buffering_reward = 0.0

        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations

        observations, reward, done, line, _, _, _, _ = env.step(action)

        history += line
        total_reward += reward
        if done:
            quality_reward = env.var1
            smooth_reward = env.var2
            buffering_reward = env.var3

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

    write2csv('evaluateRL/{}play_event'.format(round(total_reward, 1)), env.play_event)
    write2csv('evaluateRL/{}down_event'.format(round(total_reward, 1)), env.down_event)
    write2csv('evaluateRL/{}buffer_traces'.format(round(total_reward, 1)), env.buffer_size_trace)
    write2csv('evaluateRL/{}bw1'.format(round(total_reward, 1)), env.bw1_trace)
    write2csv('evaluateRL/{}bw2'.format(round(total_reward, 1)), env.bw2_trace)

    env.reset()
    # with open('history.txt', 'w+') as f:
    #     f.write(history)

    return total_reward, losses, quality_reward, smooth_reward, buffering_reward


def main():
    reward_trace = np.array([['reward_quality', 'smooth_penalty', 'rebuf_penalty', 'sum_reward']])
    # reward_trace= np.append(reward_trace,[['reward_quality', 'smooth_penalty', 'rebuf_penalty', 'sum_reward']])
    env = Env()
    gamma = 0.9
    copy_step = 25
    num_states = env.state_space()
    num_actions = env.action_space()
    hidden_units = [128, 256]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 200
    lr = 1e-4 * 0.5

    N = 3000
    total_rewards = np.array([])
    epsilon = 0.99
    decay = 0.999
    min_epsilon = 0.1
    epoch = []
    avg_rewards = []
    file_name = 'lr45_decay999_batch200_fixtrace_v15'
    current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    state = env.reset()

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    PRETRAIN = False
    if PRETRAIN:
        epsilon = 0.99
        decay = 0.99
        min_epsilon = 0.1
        N = 1000
        file_name = file_name+'_pretrain'
        TrainNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_lr4_decay999_batch200_v15.h5')
        TargetNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_lr4_decay999_batch200_v15.h5')

    test = True
    if test:
        epsilon = 0.0
        decay = 1.0
        min_epsilon = 0.0
        N = 5
        file_name = 'lr45_decay99_batch200_fixtrace_v15'
        TrainNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_{}.h5'.format(file_name))
        TargetNet.load_model(np.array([state], dtype='float32'), 'model/DQNmodel_{}.h5'.format(file_name))
        file_name = file_name+'_test'

    print('epoch', 'reward', 'quality_reward', 'smooth_reward', 'buffering_reward')
    qua_rw = []
    smth_rw = []
    rebuffering_rw =[]
    total_rewards=[]
    for n in range(N):
        epoch.append(n)
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses, quality_rw, smooth_rw, buffering_rw = play_video(env, TrainNet, TargetNet, epsilon, copy_step)
        # total_rewards = np.append(total_rewards, [total_reward])
        # avg_reward = total_rewards[max(0, n - 100):(n + 1)].mean()
        # avg_rewards.append(avg_reward)
        reward_trace = np.append(reward_trace,[quality_rw, smooth_rw, buffering_rw, total_reward])
        qua_rw.append(quality_rw)
        smth_rw.append(smth_rw)
        rebuffering_rw.append(buffering_rw)
        total_rewards.append(total_reward)


        print(n, total_rewards[-1], quality_rw, smooth_rw, buffering_rw)

        if n != 0 and n % 55 == 0:
            plt.figure(figsize=(15,3))
            plt.plot(epoch, total_rewards)
            plt.plot(epoch, avg_rewards)
            # plt.xlabel("reward")
            # plt.ylable("episode")

            plt.savefig('fig/{}.png'.format(file_name))
            plt.clf()
            f2 = 'rewards/history_{}'.format(file_name)
            np.save(f2, [total_rewards, avg_rewards])
            # write2csv('rewards', [total_rewards, avg_rewards])
            if n % 25 == 0 and not test:
                TrainNet.save_model("model/DQNmodel_{}.h5".format(file_name))


    # if test:
    #     # reward_trace =np.array(reward_trace)
    #     # write2csv('evaluateRL/rewards_track', reward_trace)
    return mean(qua_rw), mean(smth_rw), (rebuffering_rw), mean(total_rewards)

if __name__ == '__main__':
    for n in range(100):
        np.random.seed(n)
        q, s, rb, rw =main()
        print(n, q, s, rb, rw)
