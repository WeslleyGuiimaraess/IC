import gym
import tensorflow as tf
import itertools as it

from collections import deque
from time import sleep

from utils import preprocess
from train import DQNAgent, run
import matplotlib.pyplot as plt

save_model = True
load = False
skip_learning = False
watch = False

game_name = 'Enduro-v4'
model_savefolder = "./model/model"

replay_memory_size = 1000000


tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

def main():
    
    #algoritmo que realiza aprendizado (agente)
    agent = DQNAgent(load=load)

    #inicia o ambiente do jogo
    env = gym.make(game_name, render_mode='human')
    env.reset()
    
    replay_memory = deque(maxlen=replay_memory_size)

    #pega a quantidade de ações que o agente pode executar no jogo
    n = env.action_space.n
    
    #lista de ações que o agente pode executar
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    
    #print(env.render(mode='rgb_array').shape)
    if not skip_learning:
        x, y = run(agent, env, replay_memory)

        print(f'{x}\n{y}')

        if save_model:
            agent.dqn.save(model_savefolder)


    if watch:

        env = gym.make(game_name, render_mode='human')

        mean_reward = 0
        for _ in range(10):
            env.reset()

            done = False

            total_reward = 0
            for i in range(1000):
                state = preprocess(env.render(mode='rgb_array')[:155, 10:, :])
                #env.render(mode='human')
                best_action_index = agent.choose_action(state)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                observation, reward, done, info = env.step(best_action_index)

                total_reward += reward

                if done:
                    env.reset()

            # Sleep between episodes
            sleep(1.0)

            mean_reward += total_reward
            print("Total score: ", total_reward)
        
        print(f'Mean score: {mean_reward/10.0}')
    

if __name__ == '__main__':
    main()
