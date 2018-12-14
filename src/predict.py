import keras
import cv2
import numpy as np
import os
from data_generator import DataGenerator
from dicts import num2char_dict, char2num_dict

# 一次多个图片一起预测
def PredictLabels(model_for_pre, test_data_dir, img_size, downsample_factor, batch_size=64, weight_path=None):
    img_w, img_h = img_size
    img_path_list = os.listdir(test_data_dir)
    # img_path_list = img_path_list[0: 10]
    num_images = len(img_path_list)
    counter = num_images
    if weight_path is not None: # 表明传入的是一个空壳，需要加载权重参数
        model_for_pre.load_weights(weight_path, by_name=True) # by_name = True 表示按名字，只取前面一部分的权重
    predicted_labels = {}
    print("Predicting Start!")
    # 将数据装入
    for idx in range(0, num_images, batch_size):
        img_path_batch = img_path_list[idx:idx+batch_size]
        l_ipb = len(img_path_batch)
        img_batch = np.zeros((l_ipb, img_h, img_w, 1))
        # 将一个batch的图片装入内存 并进行处理
        print("There are {} images left.".format(counter))
        for i, img_path in enumerate(img_path_batch):
            img = cv2.imread(os.path.join(test_data_dir, img_path))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img, (img_w, img_h))
            gray_img = np.expand_dims(gray_img, axis=-1)
            gray_img = gray_img / 255.0 * 2.0 - 1.0 # 零中心化
            img_batch[i, :, :, :] = gray_img

        # 传输进base_net获得预测的softmax后验概率矩阵
        y_pred_probMatrix = model_for_pre.predict(img_batch)
        y_pred_length = np.full((l_ipb,), int(img_w//downsample_factor))

        # Decode 阶段 
        y_pred_labels_tensor_list, _ = keras.backend.ctc_decode(y_pred_probMatrix, y_pred_length, greedy=True) # 使用的是最简单的贪婪算法
        y_pred_labels_tensor = y_pred_labels_tensor_list[0]
        y_pred_labels = keras.backend.get_value(y_pred_labels_tensor) # 现在还是字符编码
        # 转换成字符串
        y_pred_text = ["" for _ in range(l_ipb)]
        for k in range(l_ipb):
            label = y_pred_labels[k]
            for num in label:
                if num == -1:break
                y_pred_text[k] += num2char_dict[num]
        for j in range(len(img_path_batch)):
            predicted_labels[img_path_batch[j]] = y_pred_text[j]
        counter -= batch_size

    print("Predict Finished!")
    return predicted_labels


# 使用txt文本的方式读取数据
def PredictLabels_txt(model_for_pre, test_data_dir, test_txt_path, img_size, downsample_factor, batch_size=64, weight_path=None):
    img_w, img_h = img_size
    img_path_list = os.listdir(test_data_dir)
    # 通过txt文件获取文件名
    data_txt = open(test_txt_path, "r")
    data_txt_list = data_txt.readlines()
    img_path_list = [line.split(" ")[0] for line in data_txt_list] # 所有的图片的文件名
    data_txt.close()
    num_images = len(img_path_list)
    
    counter = num_images
    
    if weight_path is not None: # 表明传入的是一个空壳，需要加载权重参数
        model_for_pre.load_weights(weight_path, by_name=True) # by_name = True 表示按名字，只取前面一部分的权重

    predicted_labels = {}
    print("Predicting Start!")
    # 将数据装入
    for idx in range(0, num_images, batch_size):
        img_path_batch = img_path_list[idx:idx+batch_size]
        l_ipb = len(img_path_batch) # 之所以不用batch_size是因为最后一个批次的数量可能小于batch_size
        img_batch = np.zeros((l_ipb, img_h, img_w, 1))
        # 将一个batch的图片装入内存 并进行处理
        print("There are {} images left.".format(counter))
        for i, img_path in enumerate(img_path_batch):
            img = cv2.imread(os.path.join(test_data_dir, img_path))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img, (img_w, img_h))
            gray_img = np.expand_dims(gray_img, axis=-1)
            gray_img = gray_img / 255.0 * 2.0 - 1.0 # 零中心化
            img_batch[i, :, :, :] = gray_img

        # 传输进base_net获得预测的softmax后验概率矩阵
        y_pred_probMatrix = model_for_pre.predict(img_batch)
        y_pred_length = np.full((l_ipb,), int(img_w//downsample_factor))

        # Decode 阶段 
        y_pred_labels_tensor_list, _ = keras.backend.ctc_decode(y_pred_probMatrix, y_pred_length, greedy=True) # 使用的是最简单的贪婪算法
        y_pred_labels_tensor = y_pred_labels_tensor_list[0]
        y_pred_labels = keras.backend.get_value(y_pred_labels_tensor) # 现在还是字符编码
        # 转换成字符串
        y_pred_text = ["" for _ in range(l_ipb)]
        for k in range(l_ipb):
            label = y_pred_labels[k]
            for num in label:
                if num == -1:break
                y_pred_text[k] += num2char_dict[num]
        for j in range(len(img_path_batch)):
            predicted_labels[img_path_batch[j]] = y_pred_text[j]
        counter -= batch_size

    print("Predict Finished!")
    return predicted_labels