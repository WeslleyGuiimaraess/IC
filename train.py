import numpy as np
import tensorflow as tf

from tqdm import trange
from time import time, sleep
from tensorflow.keras import Model, Sequential, Input, losses, metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, ReLU

import matplotlib.pyplot as plt

from utils import split_tuple, extractDigits, preprocess, get_samples

# NN learning settings (configurações de aprendizado da rede neural)
batch_size = 64

# Q-learning settings 
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 100000

# Variáveis de treinamento
num_train_epochs = 10
learning_steps_per_epoch = 500
target_net_update_steps = 10

model_savefolder = "./model/model/"

#
class DQNAgent:
    def __init__(self, num_actions=9, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995, load=False):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.optimizer = SGD(learning_rate)

        if load:
            print("Loading model from: ", model_savefolder) 
            self.dqn = tf.keras.models.load_model(model_savefolder)
        else:
            self.dqn = DQN(self.num_actions)
            self.target_net = DQN(self.num_actions)

    def update_target_net(self):
        self.target_net.set_weights(self.dqn.get_weights())
    
    #agente escolhe a ação dada a rede
    def choose_action(self, state):
        action = int(tf.argmax(self.dqn(tf.reshape(state, (1,20,30,1))), axis=1))

        return action

    #treina o agente
    def train_dqn(self, samples):
        screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(samples)

        row_ids = list(range(screen_buf.shape[0]))

        ids = extractDigits(row_ids, actions)
        done_ids = extractDigits(np.where(dones)[0])

        #aplica o algoritmo de Q learning
        with tf.GradientTape() as tape:
            tape.watch(self.dqn.trainable_variables)

            Q_prev = tf.gather_nd(self.dqn(screen_buf), ids)
            
            Q_next = self.target_net(next_screen_buf)
            Q_next = tf.gather_nd(Q_next, extractDigits(row_ids, tf.argmax(self.dqn(next_screen_buf), axis=1)))
            
            q_target = rewards + self.discount_factor * Q_next

            if len(done_ids)>0:
                done_rewards = tf.gather_nd(rewards, done_ids)
                q_target = tf.tensor_scatter_nd_update(tensor=q_target, indices=done_ids, updates=done_rewards)

            td_error = tf.keras.losses.MSE(q_target, Q_prev)

        gradients = tape.gradient(td_error, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


#rede neural
class DQN(Model):
    #define a estrutura da rede
    def __init__(self, num_actions):
        super(DQN,self).__init__()
        self.conv1 = Sequential([
                                Conv2D(8, kernel_size=6, strides=3, input_shape=(20,30,1)),
                                BatchNormalization(),
                                ReLU()
                                ])

        self.conv2 = Sequential([
                                Conv2D(8, kernel_size=3, strides=2, input_shape=(5, 9, 8)),
                                BatchNormalization(),
                                ReLU()
                                ])
        
        self.flatten = Flatten()
       
        self.state_value = Dense(1) 
        self.advantage = Dense(num_actions)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x1 = x[:, :96]
        x2 = x[:, 96:]
        x1 = self.state_value(x1)
        x2 = self.advantage(x2) 
        
        x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1,1)))
        return x


#teina o agente no ambiente(jogo)
def run(agent, env, replay_memory):
    time_start = time()
    
    x = []
    y = []
    #contabiliza a quantidade de iterações do agente dentro ambiente
    for episode in range(num_train_epochs):
        train_scores = []
        print("\nEpoch %d\n-------" % (episode + 1))

        total_reward = 0

        env.reset()

        next_screen_buf = preprocess(env.render(mode='rgb_array')[:155, 10:, :])
        action = env.action_space.sample()
        env.step(action)
        #skipa os frames para tomar a decisão
        for i in trange(learning_steps_per_epoch, leave=False):
            #trata a imagem cor/tamanho da iamgem para que possa ser utilizada
            screen_buf = next_screen_buf

            if agent.epsilon < np.random.uniform(0,1):
                action = int(tf.argmax(agent.dqn(tf.reshape(screen_buf, (1,20,30,1))), axis=1))
            else:
            # action = np.random.choice(range(self.num_actions), 1)[0]
                action = np.random.choice((1,7,8))
            #observa a ação tomado pelo agente para poder dar a recompensa



            observation, reward, done, info = env.step(action)

            #soma o valor da recompensa acumulando o total
            total_reward += reward
            #pega o proximo frame para o agente tomar a decisão
            if not done:
                next_screen_buf = preprocess(env.render(mode='rgb_array')[:155, 10:, :])
            else:
                next_screen_buf = tf.zeros(shape=screen_buf.shape)

            #caso o agente atinja o objetivo o agente reseta tudo e adiciona a recompensa 
            if done:
                train_scores.append(total_reward)
                env.reset()

            #armazena todos os dados da tela, ação, recompensa, proxima tela
            replay_memory.append((screen_buf, action, reward, next_screen_buf, done))

            #se o frame atual for maior que batch, o algoritmo começa a treinar o agente
            if i % batch_size == 0:
                agent.train_dqn(get_samples(replay_memory))
       
            #
            if ((i % target_net_update_steps) == 0):
                agent.update_target_net()
            
        print(f'Total score episode {episode}: {total_reward}')
        x.append(episode)
        y.append(total_reward)
        agent.dqn.save_weights(f'./model/model_{episode}')

        train_scores = np.array(train_scores)

    return x, y