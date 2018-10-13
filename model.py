from keras import Input, Model
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten

class CNN:
    def __init__(self, action_size):
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        input = Input((1, 200, 200))

        #  Add convolutional and normalization layers
        hidden_conv_layer_1 = Convolution2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu',
                                            data_format='channels_first')(input)
        hidden_conv_layer_2 = Convolution2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu')(
            hidden_conv_layer_1)
        hidden_conv_layer_3 = Convolution2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(
            hidden_conv_layer_2)
        # -------------------------------------------

        # make convolution layers falttend (1 dimensional)
        flattened = Flatten()(hidden_conv_layer_3)
        #  add FC layers
        hidden_fc_layer = Dense(512, activation='relu')(flattened)
        output_layer = Dense(self.action_size)(hidden_fc_layer)
        # -------------------------------------------

        # configure learning process and initialize model
        model = Model(inputs=input, output=output_layer)

        return model

