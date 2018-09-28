import gym
import random

if __name__ == "__main__":

    env = gym.make('VirtualDrone-v0')  # environment initialization

    observation = env.reset()
    for i in range(0, 1000):
        env.render()
        observation, reward, done, info = env.step(random.randint(0, 5))
        if done:
            print('Episode Finished!')
            env.reset()
    print('end')
