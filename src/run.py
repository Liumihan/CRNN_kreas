from data_generator import DataGenerator
from train import train_model
from predict import PredictLabels
from utils import check_acc
import keras
import numpy as np
import time
import vgg_blstm_ctc
import vgg_bgru_ctc
import resnet_bgru_ctc

def main():
    '''
    model_choice: 0--vgg_bgru_ctc, 1--vgg_blstm_ctc, 2--resnet_blstm_ctc
    '''
    model_dict = {0: "vgg_bgru_ctc", 1: "vgg_blstm_ctc", 2:"resnet_blstm_ctc"}
    model_choice = 0 
    model_for_train, model_for_predict = None, None

    # 各种训练时候的参数
    img_size = (128, 32) # W*H
    num_classes = 11 # 包含“blank”
    max_label_length = 12
    downsample_factor = 4
    epochs = 100

    if model_choice == 0:
        # vgg_bgru_ctc
        model_for_train = vgg_bgru_ctc.model(is_training=True, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        model_for_predict = vgg_bgru_ctc.model(is_training=False, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)       
    elif model_choice == 1:
        # vgg_blstm_ctc
        model_for_train = vgg_blstm_ctc.model(is_training=True, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        model_for_predict = vgg_blstm_ctc.model(is_training=False, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
    elif model_choice == 2:
        # resnet_blstm_ctc
        model_for_train = resnet_bgru_ctc.model(is_training=True, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
        model_for_predict = resnet_bgru_ctc.model(is_training=False, img_size=img_size, num_classes=num_classes, max_label_length=max_label_length)
    
    
    
    # 各种路径
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    weight_save_path = "../trained_weights/{}_{}_best_weight.h5".format(current_time, model_dict[model_choice])
    img_data_dir = "../data/numbers_croped"
    train_txt_path = "../data/data_txt/train.txt"
    val_txt_path = "../data/data_txt/val.txt"
    
    
    # 训练模型
    train_model(model_for_train, img_data_dir, train_txt_path, val_txt_path,
             weight_save_path, epochs=epochs, img_size=img_size, batch_size=128, 
             max_label_length=max_label_length,down_sample_factor=4)

    # 使用模型进行预测
    predict_labels = PredictLabels(model_for_predict, img_data_dir, val_txt_path, 
                                    img_size, downsample_factor, batch_size=128, 
                                    weight_path=weight_save_path)

    # check accuracy
    acc, misclassified = check_acc(predict_labels, val_txt_path)
    print("accuracy on the on the val_data is {}.".format(acc))
 
    # 保存预测的结果
    test_result_save_path = "../predicted_results/{}_acc_{:.2f}_{}_result.txt".format(current_time, acc, model_dict[model_choice])
    result_txt = open(test_result_save_path, 'w')
    result_txt.write("acc :{}.\n".format(acc))
    result_txt.write("misclassified: {}.\n".format(len(misclassified)))
    for key, value in misclassified.items():
        result_txt.write(str(key) + ": " + str(value) + '\n')
    result_txt.write("all results:\n")
    for key, value in predict_labels.items():
        result_txt.write(str(key) + ": " + str(value) + '\n')
    result_txt.close()

    return 0

if __name__=="__main__":
    main()
