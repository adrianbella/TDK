from model import CNN
from master_model import OriginalCNN
import gym
import numpy as np

if __name__ == "__main__":

    ENV_NAME = 'VirtualDrone-v0'
    env = gym.make(ENV_NAME)  # environment initialization

    database_limit = 1024
    hidden_fc_size = 8
    hidden_conv1_filters = 8
    hidden_conv2_filters = 16
    file_path = "VirtualDrone-v0_DQNAgent_conv18_conv216_fc8.h5f"
    #file_path = "VirtualDrone-v0_DDQNAgent_2018-10-15 12:16:32.416484_weights.h5f"

    model = CNN(env.action_space.n, hidden_fc_size, hidden_conv1_filters, hidden_conv2_filters, file_path)
    #model = OriginalCNN(env.action_space.n, file_path)

    observation = env.reset()

    while(1):
        env.render()
        observation = np.expand_dims(np.expand_dims(observation, axis=0), axis=0)
        prediction = model.model.predict(observation, batch_size=1)

        step = np.argmax(prediction, axis=1)

        observation, reward, done, info = env.step(step)

        print('Reward:{} '.format(reward))

        if done:
            print('--------------------------Episode finished!--------------------------')
            observation = env.reset()