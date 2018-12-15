from keras.layers import LSTM, Conv2D, Activation, MaxPooling2D, Lambda, TimeDistributed
from keras.layers import Bidirectional, Permute, Flatten, Reshape, Input, BatchNormalization
from keras.layers import Add, AveragePooling2D, GRU, Dense
from utils import ctc_loss_layer
import keras


def conv_block(input_tensor, filters, place):
    f1, f2, f3 = filters
    # main path
    x = Conv2D(f1, (1,1), name="conv2d_a" + str(place[0]))(input_tensor)
    x = BatchNormalization(name="BN_a" + str(place[0]))(x)
    x = Activation("relu", name="relu_a" + str(place[0]))(x)
    
    x = Conv2D(f2, (3,3), padding='same', name="conv2d_b" + str(place[0]))(x)
    x = BatchNormalization(name="BN_b" + str(place[0]))(x)
    x = Activation("relu", name="relu_b" + str(place[0]))(x)

    x = Conv2D(f3, (1,1), name="conv2d_c" + str(place[0]))(x)
    x = BatchNormalization(name="BN_c" + str(place[0]))(x)
    main_path = Activation("relu", name="main_path" + str(place[0]))(x)
    # short cut
    short_cut = Conv2D(f3, (1,1), name="short_cut_conv" + str(place[0]))(input_tensor)
    short_cut = BatchNormalization(name="short_cut_BN" + str(place[0]))(short_cut)
    short_cut = Activation("relu", name="short_cut" + str(place[0]))(short_cut)

    # add two path together
    out = Add(name="add" + str(place[0]))([main_path, short_cut])
    out = Activation("relu", name="out_relu" + str(place[0]))(out)
    # 更新名字的place计数变量
    place[0] += 1

    # 经过这个函数过后 tensor的 H,W 不变 channel 变成f3
    return out

def identity_block(input_tensor, filters, place):
    f1, f2, f3 = filters
    # main_path
    x = Conv2D(f1, (1, 1), name="conv2d_a" + str(place[0]))(input_tensor)
    x = BatchNormalization(name="BN_a" + str(place[0]))(x)
    x = Activation("relu", name="relu_a" + str(place[0]))(x)
    
    x = Conv2D(f2, (3, 3), padding='same', name="conv2d_b" + str(place[0]))(x)
    x = BatchNormalization(name="BN_b" + str(place[0]))(x)
    x = Activation("relu", name="relu_b" + str(place[0]))(x)

    x = Conv2D(f3, (1, 1), name="conv2d_c" + str(place[0]))(x)
    x = BatchNormalization(name="BN_c" + str(place[0]))(x)
    main_path = Activation("relu", name="main_path" + str(place[0]))(x)

    # short cut
    short_cut = input_tensor

    # add two path together
    out = Add(name="add" + str(place[0]))([main_path, short_cut])
    out = Activation("relu", name="out_relu" + str(place[0]))(out)
    # 更新名字的place计数变量
    place[0] += 1

    # 经过这个函数过后 tensor的 H,W 不变 channel 变成f3

    return out


def model(is_training, img_size=(128, 32), num_classes = 11, max_label_length=12): # resnet-50 简化版
    initializer = keras.initializers.he_normal()
    regularizer = keras.regularizers.l2(0)
    max_label_length = 12
    picture_width, picture_height = img_size
    place = [0]
    
    inputs = Input(shape=(picture_height, picture_width, 1), name='pic_inputs') # 32*128*1

    x = Conv2D(32, (3,3), padding='same', name="conv2d_0")(inputs)
    x = BatchNormalization(name="BN_0")(x)
    x = Activation("relu", name="relu_0")(x) # 32*128*1
    x = MaxPooling2D(pool_size=(2,2), name="maxpl_0")(x) # 16*32*1

    x = conv_block(x, (32, 32, 128), place) # 16*64*256
    x = identity_block(x, (32, 32, 128), place) # 16*64*256
    x = identity_block(x, (32, 32, 128), place) # 16*64*256
    
    x = MaxPooling2D(name="maxpl_1")(x) # 8*32*256

    x = conv_block(x, (128, 128, 256), place) # 8*32*512  
    x = identity_block(x, (128, 128, 256), place) # 8*32*512
    x = identity_block(x, (128, 128, 256), place) # 8*32*512
    x = identity_block(x, (128, 128, 256), place) # 8*32*512

    x = MaxPooling2D(pool_size=(2, 1), name="maxpl_2")(x) # 4*32*512

    x = conv_block(x, (128, 128, 512), place) # 4*16*1024
    x = identity_block(x, (128, 128, 512), place) # 4*16*1024
    x = identity_block(x, (128, 128, 512), place) # 4*16*1024
    x = identity_block(x, (128, 128, 512), place) # 4*16*1024
    x = identity_block(x, (128, 128, 512), place) # 4*16*1024
    x = identity_block(x, (128, 128, 512), place) # 4*16*1024

    x = MaxPooling2D(pool_size=(2, 1), name="maxpl_3")(x) # 2*32*1024

    x = conv_block(x, (256, 256, 1024), place) # 2*32*2048
    x = identity_block(x, (256, 256, 1024), place) # 2*32*2048
    x = identity_block(x, (256, 256, 1024), place) # 2*32*2048

    conv_output = AveragePooling2D(pool_size=(2,1), name="conv_output")(x) # 1*32*2048
    
    # Map2Sequence
    x = Permute((2, 3, 1), name='permute')(conv_output) # 32*2048*1
    time_stamps = TimeDistributed(Flatten(), name='FlattenByTime')(x) # 32*2048

    # RNN
    y = Bidirectional(GRU(256, return_sequences=True), name="BGRU_1")(time_stamps) # 32*2048
    y = BatchNormalization(name="BN_GRU")(y)
    y = Bidirectional(GRU(256, return_sequences=True), name="BGRU_2")(y) # 32*2048    
    y_pred = Dense(num_classes, activation="softmax", name='y_pred')(y) # 32*11

    # Transcription part (CTC_loss part)
    y_true = Input(shape=[max_label_length], name='y_true')
    y_pred_length = Input(shape=[1], name='y_pred_length')
    y_true_length = Input(shape=[1], name='y_true_length')
    ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')([y_true, y_pred, y_pred_length, y_true_length])
   
    base_model = keras.models.Model(inputs=inputs, outputs=y_pred)
    full_model = keras.models.Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    if is_training:
        return full_model
    else:
        return base_model



def main():

    # input = Input(shape=(32, 128, 256))

    # out = conv_block(input, (64,64,256), [1])
    # out = identity_block(input, (64,64,256), [0])

    # m_model = keras.models.Model(inputs=input, outputs=out)
    m_model = model(True)
    m_model.summary()

    return 0

if __name__ == "__main__":
    main()