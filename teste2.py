import gym
import numpy as np
import itertools as it
import tensorflow as tf
import skimage.color, skimage.transform
import os

from tqdm import trange
from random import sample
from time import time, sleep
from collections import deque
from tensorflow.keras import Model, Sequential, Input, losses, metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, ReLU

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000
num_train_epochs = 2
learning_steps_per_epoch = 200
target_net_update_steps = 100

# NN learning settings
batch_size = 64

# Training regime 
test_episodes_per_epoch = 100

# Other parameters
frames_per_action = 12
resolution = (40, 60)
episodes_to_watch = 20

save_model = True
load = False
skip_learning = False
watch = True


model_savefolder = "./model"


def preprocess(img):
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)
   
    return tf.stack(img)


def split_tuple(samples):
    samples = np.array(samples, dtype=object)
    screen_buf = tf.stack(samples[:,0])
    actions = samples[:,1]
    rewards = tf.stack(samples[:,2])
    next_screen_buf = tf.stack(samples[:,3])
    dones = tf.stack(samples[:,4])  
    return screen_buf, actions, rewards, next_screen_buf, dones 


def extractDigits(*argv):
    if len(argv)==1:
        return list(map(lambda x: [x], argv[0]))

    return list(map(lambda x,y: [x,y], argv[0], argv[1]))


def get_samples(memory):
    if len(memory) < batch_size:
        sample_size = len(memory)
    else:
        sample_size = batch_size

    return sample(memory, sample_size)



class DQNAgent:
    def __init__(self, num_actions=9, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995, load=load):
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
 
    def choose_action(self, state):
        if self.epsilon < np.random.uniform(0,1):
            action = int(tf.argmax(self.dqn(tf.reshape(state, (1,40,60,1))), axis=1))
        else:
            action = np.random.choice(range(self.num_actions), 1)[0]

        return action

    def train_dqn(self, samples):
        screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(samples)

        row_ids = list(range(screen_buf.shape[0]))

        ids = extractDigits(row_ids, actions)
        done_ids = extractDigits(np.where(dones)[0])

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


class DQN(Model):
    def __init__(self, num_actions):
        super(DQN,self).__init__()
        self.conv1 = Sequential([
                                Conv2D(8, kernel_size=6, strides=3, input_shape=(40,60,1)),
                                BatchNormalization(),
                                ReLU()
                                ])

        self.conv2 = Sequential([
                                Conv2D(8, kernel_size=3, strides=2, input_shape=(12, 19, 8)),
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


def run(agent, env, replay_memory):
    time_start = time()

    for episode in range(num_train_epochs):
        train_scores = []
        print("\nEpoch %d\n-------" % (episode + 1))

        total_reward = 0

        for i in trange(learning_steps_per_epoch, leave=False):
            screen_buf = preprocess(env.render(mode='rgb_array'))

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_screen_buf = screen_buf
            else:
                next_screen_buf = tf.zeros(shape=screen_buf.shape)

            if done:
                train_scores.append(total_reward)

                env.reset()

            replay_memory.append((screen_buf, action, reward, next_screen_buf, done))

            if i >= batch_size:
                agent.train_dqn(get_samples(replay_memory))
       
            if ((i % target_net_update_steps) == 0):
                agent.update_target_net()

        train_scores = np.array(train_scores)
        #print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
        #          "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())


def main():
    
    agent = DQNAgent()

    env = gym.make('Enduro-v4')
    env.reset()
    
    replay_memory = deque(maxlen=replay_memory_size)

    n = env.action_space.n
    
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    
    #print(env.render(mode='rgb_array').shape)
    if not skip_learning:
        run(agent, env, replay_memory)

        env.close()

        if save_model:
            agent.dqn.save(model_savefolder)
 
    if watch:

        env = gym.make('Enduro-v4', render_mode='human')

        mean_reward = 0
        for _ in range(5):
            env.reset()

            done = False

            total_reward = 0
            for i in range(learning_steps_per_epoch):
                state = preprocess(env.render(mode='rgb_array'))
                #env.render(mode='human')
                best_action_index = agent.choose_action(state)
                random_action = env.action_space.sample()

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                observation, reward, done, info = env.step(abs(best_action_index - 1))

                total_reward += reward

            # Sleep between episodes
            sleep(1.0)

            mean_reward += total_reward
            #print("Total score: ", total_reward)
        
        print(f'Mean score: {mean_reward/10.0}')
    

if __name__ == '__main__':
    main()