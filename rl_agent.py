import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as kls
import tensorflow.keras.layers as kl
from tensorflow.keras import Model
import tensorflow.keras.optimizers as ko
import pickle
import numpy as np
import time

# DEFAULT SETTINGS
LEARNING_RATE = 0.01
GAMMA = 0.99
A_DIM = 7
SAMPLE_SIZE = 20

class pModel(Model):

    def __init__(self):
        super().__init__()
        dense0_init = tf.keras.initializers.RandomUniform(minval=-0.125, maxval=0.125)
        self.dense0 = kl.Dense(256, activation = 'relu', kernel_initializer = dense0_init)
        out_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        self.out = kl.Dense(A_DIM, activation = 'softmax',kernel_initializer = out_init)

    def call(self, state):
        x = self.dense0(state)
        return self.out(x)
        
class aModel(Model):

    def __init__(self):
        super().__init__()
        dense0_init = tf.keras.initializers.RandomUniform(minval=-0.125, maxval=0.125)
        self.dense0 = kl.Dense(256, activation = 'relu', kernel_initializer = dense0_init)
        out_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        self.out = kl.Dense(1, activation = 'relu',kernel_initializer = out_init)

    def call(self,state):
        x = self.dense0(state)
        return self.out(x)

class vModel(Model):
    def __init__(self):
        dense0_init = tf.keras.initializers.RandomUniform(minval=-0.125, maxval=0.125)
        self.dense0 = kl.Dense(256, activation = 'relu', kernel_initializer = dense0_init)
        out_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        self.out = kl.Dense(1, activation = 'relu',kernel_initializer = out_init)

    def call(self,state):
        x = self.dense0(state)
        return self.out(x)

class policy_gradient_agent():

    def __init__(self, lr = LEARNING_RATE, gamma = GAMMA):
        self.lr = lr
        self.model = pModel()
        self.gamma = gamma
        self.model.compile(optimizer = ko.SGD(lr = lr),
                            loss = self._logits_loss)
        self.a_batch = np.array([])
        self.s_batch = np.array([])
        self.r_batch = np.array([])
        self.t_batch = np.array([])
        self.v_batch = np.array([])
        self.score_trace = np.array([])

    def _logits_loss(self, act_and_advs, logits):
        actions, advantages = tf.split(act_and_advs, 2, axis = -1)
        actions = tf.cast(actions, tf.int32)
        actions = tf.reshape(actions,[SAMPLE_SIZE])
        o_actions = tf.one_hot(actions,A_DIM)
        policy_loss = tf.math.reduce_sum(-tf.math.log(tf.math.multiply(o_actions,logits) + 1e-5) * advantages) 
                    
        return policy_loss

    def pick_action(self,dist):
        prob = np.round(dist,decimals = 4)
        action = np.random.choice(A_DIM,1 , p=prob[0])[0]
        return action

    def get_dist(self,input):
        return self.model(input)

    def add2batch(self, state, action, reward, done):
        self.sample = state
        self.a_batch = np.append(self.a_batch,action)
        if np.shape(self.s_batch)[0] == 0:
            self.s_batch = np.array([state])
        else:
            self.s_batch = np.append(self.s_batch, [state], axis = 0)
        self.r_batch = np.append(self.r_batch, reward)
        self.t_batch = np.append(self.t_batch, done)

    def calculate_value(self):
        self.v_batch = np.zeros_like(self.r_batch)
        size =  self.r_batch.size
        for i in np.flip(np.arange(size)):
            self.v_batch[i] = (self.v_batch[(i+1) % size] + self.r_batch[i]) * (1-self.t_batch[i])
        self.score_trace = np.append(self.score_trace, self.v_batch[0])

    def sample_batch(self):
        batch_size = np.shape(self.t_batch)[0]
        self.calculate_value()
        pick = np.random.choice(np.arange(batch_size-1),size = SAMPLE_SIZE)
        a_train = np.array([])
        s_train = np.array([])
        v_train = np.array([])
        r_train = np.array([])
        for i in pick:
            a_train = np.append(a_train, self.a_batch[i])
            r_train = np.append(r_train, self.r_batch[i])
            v_train = np.append(v_train, self.r_batch[i])
            if np.shape(s_train)[0] == 0:
                s_train = np.array([self.s_batch[i]])
            else:
                s_train = np.append(s_train, [self.s_batch[i]], axis = 0)
        return s_train, a_train, v_train

    def train(self):
        s_train,a_train,v_train = self.sample_batch()
        acts_and_advs = np.concatenate([a_train[:,None], v_train[:, None]], axis=-1)
        self.model.train_on_batch(s_train, acts_and_advs)
        total_reward = self.v_batch[0]
        self.reset()
        return total_reward

    def reset(self):
        self.a_batch = np.array([])
        self.s_batch = np.array([])
        self.r_batch = np.array([])
        self.t_batch = np.array([])
        self.v_batch = np.array([])

    def save_model(self):
        episode = np.size(self.score_trace)
        print(self.score_trace)
        trace_data = {
            'episode': episode,
            'score_trace': self.score_trace
        }

        self.model.save_weights("model/SGDmodel.h5")
        pickle.dump(trace_data, open('agent/SGD.pickle','wb'))

    def load_model(self):
        # trace_data = pickle.load(open('agent/SGD.pickle','wb'))
        self.model.load_weights('model/SGDmodel.h5')
        self.model.compile(optimizer = ko.SGD(lr = self.lr),
                            loss = self._logits_loss)

class actor_critic_agent():
    
    def __init__(self, lr = LEARNING_RATE, gamma = GAMMA):
        self.lr = lr
        self.actor = pModel()
        self.gamma = gamma
        self.actor.compile(optimizer = ko.SGD(lr = lr),
                            loss = self._logits_loss)
        
        self.critic = aModel()
        self.critic.compile(optimizer = ko.Adam(lr = lr * 0.1),
                            loss = kls.MSE)
        
        self.a_batch = np.array([])
        self.s_batch = np.array([])
        self.r_batch = np.array([])
        self.t_batch = np.array([])
        self.v_batch = np.array([])
        self.score_trace = np.array([])

    def _logits_loss(self, act_and_advs, logits):
        actions, advantages = tf.split(act_and_advs, 2, axis = -1)
        actions = tf.cast(actions, tf.int32)
        actions = tf.reshape(actions,[SAMPLE_SIZE])
        o_actions = tf.one_hot(actions,A_DIM)
        policy_loss = tf.math.reduce_sum(-tf.math.log(tf.math.multiply(o_actions,logits) + 1e-5) * advantages) 
                    
        return policy_loss

    def pick_action(self,dist):
        prob = np.round(dist,decimals = 4)
        action = np.random.choice(A_DIM,1 , p=prob[0])[0]
        return action

    def get_dist(self,input):
        return self.actor(input)

    def add2batch(self, state, action, reward, done):
        self.sample = state
        self.a_batch = np.append(self.a_batch,action)
        if np.shape(self.s_batch)[0] == 0:
            self.s_batch = np.array([state])
        else:
            self.s_batch = np.append(self.s_batch, [state], axis = 0)
        self.r_batch = np.append(self.r_batch, reward)
        self.t_batch = np.append(self.t_batch, done)

    def calculate_value(self):
        self.v_batch = np.zeros_like(self.r_batch)
        size =  self.r_batch.size
        for i in np.flip(np.arange(size)):
            self.v_batch[i] = (self.v_batch[(i+1) % size] + self.r_batch[i]) * (1-self.t_batch[i])
        self.score_trace = np.append(self.score_trace, self.v_batch[0])

    def sample_batch(self):
        batch_size = np.shape(self.t_batch)[0]
        self.calculate_value()
        pick = np.random.choice(np.arange(batch_size-1),size = SAMPLE_SIZE)
        a_train = np.array([])
        s_train = np.array([])
        v_train = np.array([])
        r_train = np.array([])
        for i in pick:
            a_train = np.append(a_train, self.a_batch[i])
            r_train = np.append(r_train, self.r_batch[i])
            v_train = np.append(v_train, self.r_batch[i])
            if np.shape(s_train)[0] == 0:
                s_train = np.array([self.s_batch[i]])
            else:
                s_train = np.append(s_train, [self.s_batch[i]], axis = 0)
        p_train = self.critic(s_train)
        return s_train, a_train, v_train, v_train - p_train[0]

    def train(self):
        s_train,a_train,v_train,adv = self.sample_batch()
        acts_and_advs = np.concatenate([a_train[:,None], adv[:, None]], axis=-1)
        self.actor.train_on_batch(s_train, acts_and_advs)
        self.critic.train_on_batch(s_train, v_train)
        total_reward = self.v_batch[0]
        self.reset()
        return total_reward

    def reset(self):
        self.a_batch = np.array([])
        self.s_batch = np.array([])
        self.r_batch = np.array([])
        self.t_batch = np.array([])
        self.v_batch = np.array([])

    def load_model(self):
        # trace_data = pickle.load(open('agent/SGD.pickle','wb'))
        self.model.load_weights('model/SGDmodel.h5')
        self.model.compile(optimizer = ko.SGD(lr = self.lr),
                            loss = self._logits_loss)

    def load_model(self):
        # trace_data = pickle.load(open('agent/SGD.pickle','wb'))
        self.actor.load_weights('model/A2Cmodel.h5')
        self.actor.compile(optimizer = ko.SGD(lr = self.lr),
                            loss = self._logits_loss)
