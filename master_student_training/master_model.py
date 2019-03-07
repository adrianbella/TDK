from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras.models import Sequential


class MasterCNN:
    def __init__(self, action_size, file_path):
        self.action_size = action_size
        self.file_path = file_path
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu',
                         data_format='channels_first', input_shape=(1, 200, 200)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))

        # make convolution layers falttend (1 dimensional)
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))

        try:
            model.load_weights(filepath=self.file_path)
            print('Loaded master_weights was successful')
        except ImportError:
            print('Loaded master_weights aborted! File not found:{} '.format(self.file_path))

        return model
