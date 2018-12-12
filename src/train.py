from utils import fake_ctc_loss
import keras
import os
import cv2
import time
from data_generator import DataGenerator

def train_model(model, train_data_dir, val_data_dir, save_path, img_size=(128,32), batch_size=128, max_label_length=12, down_sample_factor=4, epochs=100):
    print("Training start!")
    model_save_path, weight_save_path = save_path
    #callbacks  
    save_model_cbk = keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    save_weights_cbk = keras.callbacks.ModelCheckpoint(weight_save_path, save_best_only=True, save_weights_only=True)
    early_stop_cbk = keras.callbacks.EarlyStopping(patience=10)
    reduce_lr_cbk = keras.callbacks.ReduceLROnPlateau(patience=5)
    
    # compile
    model.compile(optimizer='adam', loss={'ctc_loss_output': fake_ctc_loss})
    # fit_generator
    train_gen = DataGenerator(train_data_dir, img_size, down_sample_factor, batch_size, max_label_length)
    val_gen = DataGenerator(val_data_dir, img_size, down_sample_factor, batch_size, max_label_length)
    model.fit_generator(generator=train_gen.get_data(),
                        steps_per_epoch=train_gen.img_number//batch_size,
                        validation_data=val_gen.get_data(),
                        validation_steps=val_gen.img_number//batch_size,
                        callbacks=[save_weights_cbk, early_stop_cbk, save_model_cbk], 
                        epochs = epochs)
    print("Training finished!")
    return 0



def main():
    # model_for_train = CRNN_model(is_training=True)
    # model_for_predict = CRNN_model(is_training=False)
    im_size = (128, 32) # W*H
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    model_save_path = "../model/" + current_time + "_best_model.h5"
    weight_save_path = "../model/weights/" + current_time + "_best_weight.h5"
    train_data_path = "../data/numbers_training_croped"
    val_data_path = "../data/numbers_val_croped"
    # data_dir_for_predict = "../data/small_test_croped"
    data_dir_for_predict = train_data_path
    test_result_save_path = "../data/result/" + current_time + "_result.txt"
    save_path = (model_save_path, weight_save_path)
    train_model(model_for_train, train_data_path, val_data_path, save_path, (128, 32))

    # weight_save_path = "../model/weights/2018_12_09_20_45_26_best_weight.h5"
    # predict_label = PredictLabels(model_for_predict, weight_save_path, data_dir_for_predict, im_size)

    result_txt = open(test_result_save_path, 'w')
    for key, value in predict_label.items():
        result_txt.write(str(key) + ": " + str(value) + '\n')
    result_txt.close()
    return 0

if __name__ == "__main__":
    main()
