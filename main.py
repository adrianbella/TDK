import env
import random

if __name__ == "__main__":

    env = env.Environment()  # environment initialization

    observation = env.get_random_initial_state()

    for i in range(0, 100):
        reward, observation = env.step(random.randint(0, 5))

    print('end')
