import logging
from configparser import ConfigParser
import datetime
import os
import keras
import csv
import numpy as np


class Logger(keras.callbacks.Callback):
    def __init__(self, section, ENV_NAME):

        self.section = section
        self.ENV_NAME = ENV_NAME
        self.config = ConfigParser()
        self.directory = './log/'
        self.config_file = './config.ini'
        self.file_path = './training_weights/'
        self.step_count = 0

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        logging.basicConfig(
            filename=self.directory + self.ENV_NAME + '_' + self.section + '_' + str(datetime.datetime.now()) + '.log',
            level=logging.DEBUG)

        try:
            self.log_config_parameters(self.config_file)
        except IOError:
            print('File config.ini doesn\'t exist!')

    def log_config_parameters(self, file):
        dict1 = {}
        self.config.read(file)
        options = self.config.options(self.section)
        logging.info("Section: {}".format(self.section))
        for option in options:
            try:
                logging.info("\t\t {} : {}".format(option, self.config.get(self.section, option)))
            except:
                dict1[option] = None

    def log_model_architect(self, model):
        model.summary(print_fn=logging.info)

    def set_model(self, model):
        self.model = model

    def log_history(self, history):
        logging.info('History:')
        logging.info('Episod average rewards: {}'.format(history.history.get('episode_reward')))
        logging.info('Number of episode steps: {}'.format(history.history.get('nb_episode_steps')))
        logging.info('Number of steps: {}'.format(history.history.get('nb_steps')))

    def on_train_begin(self, logs={}):
        self.rewards = []
        self.actions = []
        self.episode = []
        self.losses = []
        self.q_values = []
        self.epsilone = []
        self.model.save_weights(self.file_path + self.ENV_NAME + '_' + self.section + '_' + str(self.step_count) + '_' + str(datetime.datetime.now()))
        print('training_begin')

    def on_batch_end(self, batch, logs={}):
        self.step_count += 1

        if logs.get('episode') > 25 and (
                    self.section == 'DQNAgent' or self.section == 'DDQNAgent' or self.section == 'SARSAAgent' or self.section == 'minimizedDQN')\
                    and not (np.isnan(logs.get('metrics')[0]) and np.isnan(logs.get('metrics')[2]) and np.isnan(logs.get('metrics')[3])):
            self.rewards.append(logs.get('reward'))
            self.actions.append(logs.get('action'))
            self.episode.append(logs.get('episode'))
            self.losses.append(logs.get('metrics')[0])
            self.q_values.append(logs.get('metrics')[2])
            self.epsilone.append(logs.get('metrics')[3])
            if (self.step_count % 10000 == 0):
                self.model.save_weights(self.file_path + self.ENV_NAME + '_' + self.section + '_' + str(self.step_count) + '_' + str(datetime.datetime.now()))

    def on_train_end(self, logs={}):
        if len(self.rewards) > 1:
            with open(self.directory + self.ENV_NAME + '_' + self.section + '_' + str(datetime.datetime.now()) + '.csv',
                      'w+') as csvfile:
                fieldnames = ['reward', 'actions', 'episodes', 'losses', 'q_values', 'epsilone']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.rewards)):
                    writer.writerow({'reward': self.rewards[i], 'actions': self.actions[i], 'episodes': self.episode[i],
                                     'losses': self.losses[i],
                                     'q_values': self.q_values[i], 'epsilone': self.epsilone[i]})

        self.rewards.clear()
        self.actions.clear()
        self.episode.clear()
        self.losses.clear()
        self.q_values.clear()
        self.epsilone.clear()
        print('training_end')
