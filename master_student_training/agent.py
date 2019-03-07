import keras

import numpy as np


class Agent:
    def __init__(self, action_size, database_limit, file_path):
        self.action_size = action_size
        self.database_limit = database_limit
        self.file_path = './student_weights/' + file_path

    def fit_student(self, master, student, env, logger):

        labels = np.zeros((self.database_limit, 1), dtype=np.float32)
        observations = np.zeros((self.database_limit, 1, 200, 200), dtype=np.float32)

        for j in range(0, 100000):

            for i in range(0, self.database_limit):
                observation = env.reset()
                observation = np.expand_dims(np.expand_dims(observation, axis=0), axis=0)

                q_values = master.model.predict(observation, batch_size=1)

                labels[i] = np.argmax(q_values, axis=1)
                observations[i][0] = observation[0][0]

            one_hot_labels = keras.utils.to_categorical(labels, self.action_size)
            student.model.fit(observations, one_hot_labels, epochs=1, batch_size=64, callbacks=[logger], verbose=2)

            if j % 50 == 0:
                student.model.save_weights(filepath=self.file_path + '.h5f', overwrite=True)
