from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras.models import Sequential


class MinimizedCNN:
    def __init__(self, action_size, hidden_fc_size, hidden_conv1_filters, hidden_conv2_filters, file_path):
        self.action_size = action_size
        self.hidden_fc_size = hidden_fc_size
        self.hidden_conv1_filters = hidden_conv1_filters
        self.hidden_conv2_filters = hidden_conv2_filters
        self.file_path = file_path
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu',
                         data_format='channels_first', input_shape=(1, 200, 200)))
        model.add(Conv2D(self.hidden_conv1_filters, (1, 1), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Conv2D(self.hidden_conv2_filters, (1, 1), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))

        # make convolution layers falttend (1 dimensional)
        model.add(Flatten())

        model.add(Dense(self.hidden_fc_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))

        try:
            model.load_weights(filepath=self.file_path)
            print('Loaded master_weights was successful')
        except ImportError:
            print('Loaded master_weights aborted! File not found:{} '.format(self.file_path))

        return model
