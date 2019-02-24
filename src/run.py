from data_generator import DataGenerator
from train import train_model
from predict import PredictLabels, PredictLabels_by_filename
from utils import check_acc
import keras
import numpy as np
import time
import vgg_blstm_ctc
import vgg_bgru_ctc
import resnet_bgru_ctc
import resnet18_blstm

# 选择显卡以及 控制显存的占用
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)

def main():
    '''
    model_choice: 0--vgg_bgru_ctc, 1--vgg_blstm_ctc, 2--resnet_blstm_ctc 3--resnet18_blstm
    '''
    model_dict = {0: "vgg_bgru_ctc", 1: "vgg_blstm_ctc", 2:"resnet_blstm_ctc", 3:'resnet18_blstm'}
    model_choice = 1
    model_for_train, model_for_predict = None, None

    
    
    
    # 各种路径 以及参数
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    # weight_save_path = "../trained_weights/{}_{}_best_weight.h5".format(current_time, model_dict[model_choice])
    weight_save_path = "../trained_weights/300wbest_vgg_blstm_ctc_best_weight.h5"
    # 数字训练路径
    # img_data_dir = "../data/numbers_croped/img"
    # train_txt_path = "../data/numbers_croped/txt/train.txt"
    # val_txt_path = "../data/numbers_croped/txt/test.txt"
    # img_size = (128, 32) # W*H
    # num_classes = 11 # 包含“blank”
    # max_label_length = 12
    # epochs = 100

    # img_data_dir = "../data/all_data_croped"
    # train_txt_path = "../data/data_txt/all_except_long/train_data.txt"
    # val_txt_path = "../data/data_txt/all_except_long/test_data.txt"

    # # 300w+ 训练路径
    img_data_dir = "../data/img_300w/img"
    train_txt_path = "../data/img_300w/txt/train.txt"
    val_txt_path = "../data/img_300w/txt/test.txt"

    # part 300w+ 训练参数
    #路径
    # img_data_dir = "../data/part_300w/img"
    # train_txt_path = "../data/part_300w/txt/train.txt"
    # val_txt_path = "../data/part_300w/txt/train.txt"
    img_size = (280, 32) # W*H
    num_classes = 5991 # 把最后以为当成"blank"， 舍弃掉第一位
    max_label_length = 10
    epochs = 100

    if model_choice == 0:
        # vgg_bgru_ctc
        model_for_train = vgg_bgru_ctc.model(is_training=True, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        model_for_predict = vgg_bgru_ctc.model(is_training=False, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        downsample_factor = 4       
    elif model_choice == 1:
        # vgg_blstm_ctc
        model_for_train = vgg_blstm_ctc.model(is_training=True, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        model_for_predict = vgg_blstm_ctc.model(is_training=False, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        downsample_factor = 4
    elif model_choice == 2:
        # resnet_blstm_ctc
        model_for_train = resnet_bgru_ctc.model(is_training=True, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        model_for_predict = resnet_bgru_ctc.model(is_training=False, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        downsample_factor = 4
    elif model_choice == 3:
        # resnet18_blstm
        model_for_train = resnet18_blstm.model(is_training=True, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        model_for_predict = resnet18_blstm.model(is_training=False, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        downsample_factor = 8
    
    
    # 训练模型
    # train_model(model_for_train, img_data_dir, train_txt_path, val_txt_path,
    #          weight_save_path, epochs=epochs, img_size=img_size, batch_size=64, 
    #          max_label_length=max_label_length,down_sample_factor=downsample_factor)

    # # 使用模型进行预测
    # predict_labels = PredictLabels(model_for_predict, img_data_dir, val_txt_path, 
    #                                 img_size, downsample_factor, batch_size=128, 
    #                                 weight_path=weight_save_path)
    img_data_dir = '../data/btn/processed_img'
    predict_labels = PredictLabels_by_filename(model_for_predict, img_data_dir, img_size, downsample_factor, weight_path=weight_save_path)
    # check accuracy
    # acc, misclassified = check_acc(predict_labels, val_txt_path)
    # print("accuracy on the on the val_data is {}.".format(acc))
 
    # 保存预测的结果
    # test_result_save_path = "../predicted_results/{}_acc_{:.2f}_{}_result.txt".format(current_time, acc, model_dict[model_choice])
    test_result_save_path = "../predicted_results/{}_{}_result.txt".format(current_time, model_dict[model_choice])
    result_txt = open(test_result_save_path, 'w')
    # result_txt.write("acc :{}.\n".format(acc))
    # result_txt.write("misclassified: {}.\n".format(len(misclassified)))
    # for key, value in misclassified.items():
    #     result_txt.write(str(key) + ": " + str(value) + '\n')
    # result_txt.write("all results:\n")

    for key, value in predict_labels.items():
        result_txt.write(str(key) + ": " + str(value) + '\n')
    result_txt.close()

    return 0

if __name__=="__main__":
    main()
