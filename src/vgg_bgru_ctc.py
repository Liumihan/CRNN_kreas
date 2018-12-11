import keras
from keras.layers import Lambda, Dense, Bidirectional, GRU, Flatten, TimeDistributed, Permute, Activation, Input, Reshape, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras import backend as K
import numpy as np
import os
import tensorflow as tf
from data_generator import *
from utils import ctc_loss_layer

def model(is_training):

    initializer = keras.initializers.he_normal()
    regularizer = keras.regularizers.l2(0)
    max_label_length = 12


    picture_width = 128
    picture_height = 32
    #CNN part
    inputs = Input(shape=(picture_height, picture_width, 1), name='pic_inputs') # H×W×1 32*128*1
    x = Conv2D(64, (3,3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizer, 
                kernel_initializer=initializer, use_bias=True, name='conv2d_1')(inputs) # 32*128*64 
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='maxpl_1')(x) # 16*64*64

    x = Conv2D(128, (3,3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizer, # 16*64*128
                kernel_initializer=initializer, use_bias=True, name='conv2d_2')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='maxpl_2')(x) # 8*32*128
    
    x = Conv2D(256, (3,3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizer, # 8*32*256
                kernel_initializer=initializer, use_bias=True, name='conv2d_3')(x) 
    x = Conv2D(256, (3,3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizer, # 8*32*256
                kernel_initializer=initializer, use_bias=True, name='conv2d_4')(x)
    x = MaxPooling2D(pool_size=(2,1), strides=(2,1), name='maxpl_3')(x) # 4*32*256

    x = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_regularizer=regularizer, # 4*32*512
                kernel_initializer=initializer, use_bias=True, name='conv2d_5')(x)
    x = BatchNormalization(axis=-1, name='BN_1')(x)
    x = Activation("relu", name='relu_1')(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_regularizer=regularizer, # 4*32*512
                kernel_initializer=initializer, use_bias=True, name='conv2d_6')(x)
    x = BatchNormalization(axis=-1, name='BN_2')(x)
    x = Activation("relu", name='relu_2')(x)

    x = MaxPooling2D(pool_size=(2,1), strides=(2,1), name='maxpl_4')(x) # 2*32*512
    x = Conv2D(512, (2,2), strides=(1,1), padding='same', kernel_regularizer=regularizer, # 2*32*512
                activation='relu', kernel_initializer=initializer, use_bias=True, name='conv2d_7')(x)
    conv_otput = MaxPooling2D(pool_size=(2, 1), name="conv_output")(x) # 1*32*512
    
    # Map2Sequence part
    rnn_input = Permute((2, 3, 1), name='permute')(conv_otput) # 32*512*1
    rnn_output = TimeDistributed(Flatten(), name='for_flatten_by_time')(rnn_input) # 32*512

    # RNN part
    y = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=regularizer), name='GRU_1')(rnn_output) # 32*512
    y = BatchNormalization(name='BN_3')(y)
    y = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=regularizer), name='GRU_2')(y) # 32*512

                                                                        # 尝试跳过rnn层
    y_pred = Dense(11, activation='softmax', name='y_pred', kernel_regularizer=regularizer)(y) # 32*11 这用来做evaluation 和 之后的test检测
                                            # 在backend的实现ctc_loss的时候没有执行softmax操作所以这里必须要在使用softmax!!!!
    base_model = keras.models.Model(inputs=inputs, outputs=y_pred)
    print('BASE_MODEL: ')
    base_model.summary()


    # Transcription part (CTC_loss part)
    y_true = Input(shape=[max_label_length], name='y_true')
    y_pred_length = Input(shape=[1], name='y_pred_length')
    y_true_length = Input(shape=[1], name='y_true_length')

    ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')([y_true, y_pred, y_pred_length, y_true_length])

    model = keras.models.Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    print("FULL_MODEL: ")
    model.summary()
    if is_training:
        return model
    else:
        return base_model



