import env
import random
import numpy as np

from display import Display

if __name__ == "__main__":

    env = env.Environment()  # environment initialization

    observation = env.get_random_initial_state()
    Display.show_img(observation)
    for i in range(0, 1000):
        reward, observation = env.step(random.randint(0, 5))
        Display.show_img(observation)
    print('end')
