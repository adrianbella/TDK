import gym
import time

from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

import numpy as np


def _build_master_model(action_size, file_path):
    model = Sequential()

    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu',
                     data_format='channels_first', input_shape=(1, 200, 200)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))

    # make convolution layers falttend (1 dimensional)
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size, activation='softmax'))

    try:
        model.load_weights(filepath=file_path)
        print('Loaded master_weights was successful')
    except ImportError:
        print('Loaded master_weights aborted! File not found:{} '.format(file_path))

    return model


def _build_student_model(action_size, file_path, hidden_fc_size, hidden_conv1_filters, hidden_conv2_filters):
    model = Sequential()

    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu',
                     data_format='channels_first', input_shape=(1, 200, 200)))
    model.add(Conv2D(hidden_conv1_filters, (1, 1), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(hidden_conv2_filters, (1, 1), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))

    # make convolution layers falttend (1 dimensional)
    model.add(Flatten())

    model.add(Dense(hidden_fc_size, activation='relu'))
    model.add(Dense(action_size, activation='softmax'))

    try:
        model.load_weights(filepath=file_path)
        print('Loaded student_weights was successful')
    except ImportError:
        print('Loaded student_weights aborted! File not found:{} '.format(file_path))

    return model

if __name__ == '__main__':

    ENV_NAME = 'VirtualDrone-v0'
    env = gym.make(ENV_NAME)  # environment initialization

    action_size = env.action_space.n
    master_file_path = './master_weight.h5f'
    student_file_path = './student_weight.h5f'
    hidden_fc_size = 8
    hidden_conv1_filters = 8
    hidden_conv2_filters = 16

    master_model = _build_master_model(action_size, master_file_path)
    student_model = _build_student_model(action_size, student_file_path, hidden_fc_size, hidden_conv1_filters, hidden_conv2_filters)

    avg_master_execution_time = 0
    avg_student_execution_time = 0

    test_iterations = 100

    for i in range(0, test_iterations):
        observation = env.reset()
        observation = np.expand_dims(np.expand_dims(observation, axis=0), axis=0)

        #start master timer
        master_start_time = time.time()
        master_model.predict(observation, batch_size=1)
        #stop master timer
        master_finish_time = time.time()
        avg_master_execution_time += master_finish_time - master_start_time

        # start student timer
        student_start_time = time.time()
        student_model.predict(observation, batch_size=1)
        # stop student timer
        student_finish_time = time.time()
        avg_student_execution_time += student_finish_time - student_start_time

    print('Master networks average feed forward time: {} [ms] after {} iterations'.format(
        (avg_master_execution_time / test_iterations) * 1000, test_iterations))
    print('Student networks average feed forward time: {} [ms] after {} iterations'.format(
        (avg_student_execution_time / test_iterations) * 1000, test_iterations))
