import logging
from configparser import ConfigParser
import datetime
import os
import keras


class Logger(keras.callbacks.Callback):
    def __init__(self, section):

        self.section = section
        self.config = ConfigParser()
        directory = './log/'
        config_file = './config.ini'
        self.rewards = []
        self.actions = []

        if not os.path.exists(directory):
            os.makedirs(directory)

        logging.basicConfig(filename=directory + str(datetime.datetime.now()) + '.log', level=logging.DEBUG)

        try:
            self.log_config_parameters(config_file)
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
        print('training_begin')

    def on_batch_end(self, batch, logs={}):
        self.rewards.append(logs.get('reward'))
        self.actions.append(logs.get('action'))
        self.episode.append(logs.get('episode'))
        self.losses.append(logs.get('loss'))

    def on_train_end(self, logs={}):
        for i in range(len(self.rewards)):
            logging.info("Reward:{}, actions:{}, episode:{}, loss: {}".format(self.rewards[i], self.actions[i], self.episode[i], self.losses[i]))
        self.rewards.clear()
        self.actions.clear()
        self.losses.clear()
        print('training_end')
