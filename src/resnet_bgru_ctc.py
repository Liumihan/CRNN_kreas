from keras.layers import LSTM, Conv2D, Activation, MaxPooling2D, Lambda, TimeDistributed
from keras.layers import Bidirectional, Permute, Flatten, Reshape, Input, BatchNormalization

def model(is_training):
    initializer = keras.initializers.he_normal()
    regularizer = keras.regularizers.l2(0)
    max_label_length = 12
    picture_width = 128
    picture_height = 32
    
    input = Input(shape=(picture_height, picture_width, 1), name="pic_input")

    x = Conv2D(64, (7, 7), strides = 2, padding='same', use_bias=True, )(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')