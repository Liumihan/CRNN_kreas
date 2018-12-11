from network import CRNN_model
from data_generator import DataGenerator
from train import train_model
from predict import PredictLabel, PredictLabels
import keras
import numpy as np
import time

def main():
    
    # 获取模型
    model_for_train = CRNN_model(is_training=True)
    model_for_predict = CRNN_model(is_training=False)
    # 各种训练时候的参数
    img_size = (128, 32) # W*H
    downsample_factor = 4
    epochs = 100
    # 各种路径
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    model_save_path = "../model/" + current_time + "_best_model.h5"
    weight_save_path = "../model/weights/" + current_time + "_best_weight.h5"
    # weight_save_path = "../model/weights/2018_12_09_20_45_26_best_weight.h5"
    train_data_path = "../data/numbers_training_croped"
    val_data_path = "../data/numbers_val_croped"
    # data_dir_for_predict = "../data/small_test_croped"
    data_dir_for_predict = val_data_path
    save_path = (model_save_path, weight_save_path)
    
    # 训练模型
    train_model(model_for_train, train_data_path, val_data_path, save_path, img_size, epochs=epochs)

    # 使用训练好的模型进行预测
    # predict_label = PredictLabel(model_for_predict, weight_save_path, data_dir_for_predict, img_size, downsample_factor)
    predict_labels = PredictLabels(model_for_predict, weight_save_path, data_dir_for_predict, img_size, downsample_factor, batch_size=128)

    # check accuracy
    acc = 0
    for gt, pre in predict_labels.items():
        if gt.split("_")[0] == pre:
            acc += 1
    acc /= len(predict_labels)
    print("accuracy on the on the val_data is {}.".format(acc))

    # 保存预测的结果
    test_result_save_path = "../data/result/" + current_time + "acc:{:.2f}".format(acc) + "_result.txt"
    result_txt = open(test_result_save_path, 'w')
    result_txt.write("acc :{}.\n".format(acc))
    for key, value in predict_labels.items():
        result_txt.write(str(key) + ": " + str(value) + '\n')
    result_txt.close()

    return 0

if __name__=="__main__":
    main()
