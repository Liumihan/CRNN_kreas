import keras
from keras.layers import Lambda, Dense, Bidirectional, GRU, Flatten, TimeDistributed, Permute, Activation, Input
from keras.layers import LSTM, Reshape, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.layers import ZeroPadding2D, AveragePooling2D, Add
from keras import backend as K
import numpy as np
import os
import tensorflow as tf
from data_generator import *
from utils import ctc_loss_layer


def model(is_training=True, img_size=(280, 32), num_classes=5991, max_label_length=10):
    img_width, img_height = img_size
    pic_inputs = Input(shape=(img_height, img_width, 1), name='pic_inputs')
    initializer = keras.initializers.he_normal()

    x = ZeroPadding2D(padding=(2,2), name='pad1')(pic_inputs)
    x = Conv2D(64, (5, 5), strides=(2, 2), name='conv1', kernel_initializer=initializer)(x)
    x = BatchNormalization(name='BN1')(x)
    x = Activation('relu', name='relu1')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpl1')(x)

    # res2a
    conv_branch = ZeroPadding2D(padding=(1,1), name='res2a_pad1')(x)
    conv_branch = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='res2a_conv1', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res2a_BN1')(conv_branch)
    conv_branch = Activation('relu', name='res2a_relu1')(conv_branch)
    conv_branch = ZeroPadding2D(padding=(1,1), name='res2a_pad2')(conv_branch)
    conv_branch = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, name='res2a_conv2', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res2a_BN2')(conv_branch)
    # short cut
    short_cut = Conv2D(64, (1, 1), strides=(1, 1), name='res2a_sc', kernel_initializer=initializer)(x)
    short_cut = BatchNormalization(name='res2a_sc_BN')(short_cut)
    # add
    x = Add(name='res2a_add')([short_cut, conv_branch])
    x = Activation('relu', name='res2a_add_relu')(x)
    # res2b
    conv_branch = ZeroPadding2D(padding=(1,1), name='res2b_pad1')(x)
    conv_branch = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='res2b_conv1', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res2b_BN1')(conv_branch)
    conv_branch = Activation('relu', name='res2b_relu1')(conv_branch)
    conv_branch = ZeroPadding2D(padding=(1,1), name='res2b_pad2')(conv_branch)
    conv_branch = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, name='res2b_conv2', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res2b_BN2')(conv_branch)
    # short cut
    # 就是x
    # add
    x = Add(name='res2b_add')([x, conv_branch])
    x = Activation('relu', name='res2b_add_relu')(x)

    # res3a
    conv_branch = ZeroPadding2D(padding=(1,1), name='res3a_pad1')(x)
    conv_branch = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='res3a_conv1', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res3a_BN1')(conv_branch)
    conv_branch = Activation('relu', name='res3a_relu1')(conv_branch)
    conv_branch = ZeroPadding2D(padding=(1,1), name='res3a_pad2')(conv_branch)
    conv_branch = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, name='res3a_conv2', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res3a_BN2')(conv_branch)
    # short cut
    short_cut = Conv2D(64, (1, 1), strides=(2, 2), name='res3a_sc', kernel_initializer=initializer)(x)
    short_cut = BatchNormalization(name='res3a_sc_BN')(short_cut)
    # add
    x = Add(name='res3a_add')([short_cut, conv_branch])
    x = Activation('relu', name='res3a_add_relu')(x)
    # res3b
    conv_branch = ZeroPadding2D(padding=(1,1), name='res3b_pad1')(x)
    conv_branch = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, name='res3b_conv1', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res3b_BN1')(conv_branch)
    conv_branch = Activation('relu', name='res3b_relu1')(conv_branch)
    conv_branch = ZeroPadding2D(padding=(1,1), name='res3b_pad2')(conv_branch)
    conv_branch = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, name='res3b_conv2', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res3b_BN2')(conv_branch)
    # short cut
    # 就是x
    # add
    x = Add(name='res3b_add')([x, conv_branch])
    x = Activation('relu', name='res3b_add_relu')(x)

    # res4a
    conv_branch = ZeroPadding2D(padding=(1,1), name='res4a_pad1')(x)
    conv_branch = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, name='res4a_conv1', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res4a_BN1')(conv_branch)
    conv_branch = Activation('relu', name='res4a_relu1')(conv_branch)
    conv_branch = ZeroPadding2D(padding=(1,1), name='res4a_pad2')(conv_branch)
    conv_branch = Conv2D(128, (3, 3), strides=(1, 1), use_bias=False, name='res4a_conv2', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res4a_BN2')(conv_branch)
    # short cut
    short_cut = Conv2D(128, (1, 1), strides=(1, 1), name='res4a_sc', kernel_initializer=initializer)(x)
    short_cut = BatchNormalization(name='res4a_sc_BN')(short_cut)
    # add
    x = Add(name='res4a_add')([short_cut, conv_branch])
    x = Activation('relu', name='res4a_add_relu')(x)
    # res4b
    conv_branch = ZeroPadding2D(padding=(1,1), name='res4b_pad1')(x)
    conv_branch = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, name='res4b_conv1', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res4b_BN1')(conv_branch)
    conv_branch = Activation('relu', name='res4b_relu1')(conv_branch)
    conv_branch = ZeroPadding2D(padding=(1,1), name='res4b_pad2')(conv_branch)
    conv_branch = Conv2D(128, (3, 3), strides=(1, 1), use_bias=False, name='res4b_conv2', kernel_initializer=initializer)(conv_branch)
    conv_branch = BatchNormalization(name='res4b_BN2')(conv_branch)
    # short cut
    # 就是x
    # add
    x = Add(name='res4b_add')([x, conv_branch])
    x = Activation('relu', name='res4b_add_relu')(x)
    conv_output = AveragePooling2D(pool_size=(4, 1), strides=(1, 1), name='conv_output')(x)

    # Map2Sequence part
    conv_output = Permute((2,3,1), name='permute')(conv_output)
    rnn_input = TimeDistributed(Flatten(), name='for_flatten_by_name')(conv_output)

    # RNN part
    y = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer=initializer), name='BLSTM1')(rnn_input)
    y = BatchNormalization(name='rnn_BN')(y)
    y = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer=initializer), name='BLSTM2')(y)

    y_pred = Dense(num_classes, activation='softmax', name='y_pred', kernel_initializer=initializer)(y)

    # Transcription part (CTC_loss part)
    y_true = Input(shape=[max_label_length], name='y_true')
    y_pred_length = Input(shape=[1], name='y_pred_length')
    y_true_length = Input(shape=[1], name='y_true_length')
    

    ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')([y_true, y_pred, y_pred_length, y_true_length])
    base_model = keras.models.Model(inputs=pic_inputs, outputs=y_pred)
    full_model = keras.models.Model(inputs=[y_true, pic_inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    print('base_model:')
    base_model.summary()
    print('full_model:')
    full_model.summary()
    if is_training:
        return full_model
    else:
        return base_model

def main():
    model()

    return 0

if __name__ == '__main__':
    main()


    