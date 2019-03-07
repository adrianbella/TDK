import logging
import os
import keras
import csv


class Logger(keras.callbacks.Callback):
    def __init__(self, file_path, student_model):

        self.directory = './log/'
        weights = './student_weights/'
        self.file_path = self.directory + file_path

        if not os.path.exists(weights):
            os.makedirs(weights)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        logging.basicConfig(filename=self.file_path + '.log', level=logging.DEBUG)

        student_model.model.summary(print_fn=logging.info)

        with open(self.file_path + '.csv', 'w+') as csvfile:
            fieldnames = ['acc', 'batch', 'loss', 'size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def on_train_begin(self, logs={}):
        self.acc = []
        self.batch = []
        self.loss = []
        self.size = []
        print('training_begin')

    def on_batch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.batch.append(logs.get('batch'))
        self.loss.append(logs.get('loss'))
        self.size.append(logs.get('size'))
        logging.info('acc:{}, loss:{}, batch{}'.format(logs.get('acc'), logs.get('loss'), logs.get('batch')))

    def on_train_end(self, logs={}):
        if len(self.acc) > 1:
            with open(self.file_path + '.csv', 'a+') as csvfile:
                fieldnames = ['acc', 'batch', 'loss', 'size']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for i in range(len(self.acc)):
                    writer.writerow({'acc': self.acc[i], 'batch': self.batch[i], 'loss': self.loss[i],
                                     'size': self.size[i]})

        self.acc.clear()
        self.batch.clear()
        self.loss.clear()
        self.size.clear()
        print('training_end')
