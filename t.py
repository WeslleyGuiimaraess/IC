import gym
import cv2 as cv


env = gym.make('Enduro-v4')
env.reset()

for i in range(10000):
    screen = env.render(mode='rgb_array')[:155, 10:, ::-1]
    #env.render(mode='human')
    action = env.action_space.sample()

    # Instead of make_action(a, frame_repeat) in order to make the animation smooth
    observation, reward, done, info = env.step(action)

    cv.imshow('teste', screen)
    cv.waitKey(1)