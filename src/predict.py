import keras
import cv2
import numpy as np
import os
from data_generator import DataGenerator
from dicts import num2char_dict, char2num_dict


# 每次一个图片来做predict
def PredictLabel(model_for_pre, weight_path, test_data_dir, img_size, downsample_factor):
    img_w, img_h = img_size
    img_path_list = os.listdir(test_data_dir)
    num_images = len(img_path_list)
    model_for_pre.load_weights(weight_path, by_name=True) # by_name = True 表示只取前面一部分的权重
    predict_label = {}
    print("start predicting! There are total {} images!".format(len(img_path_list)))
    for img_path in img_path_list:
        print("Predicting image {0}. {1} images left!".format(img_path, num_images))
        img = cv2.imread(os.path.join(test_data_dir, img_path))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # gau = cv2.GaussianBlur(gray_img, (3, 3), 1.5)
        # gau_closing = cv2.morphologyEx(gau, cv2.MORPH_CLOSE, (5, 5))
        # max_gau_closing = np.where(gau_closing < 240, np.zeros(gau_closing.shape), np.ones(gau_closing.shape)*255)

        gray_img = cv2.resize(gray_img, (img_w, img_h))

        gray_img = np.expand_dims(gray_img, axis=-1)
        gray_img = np.expand_dims(gray_img, axis=0)

        gray_img = gray_img / 255.0 * 2.0 - 1.0 # 零中心化
        
        y_pred = model_for_pre.predict(gray_img)
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        y_pred_length = np.full((1,), int(img_w//downsample_factor))
        # 解码阶段
        y_pred_label_tensor_list, _ = keras.backend.ctc_decode(y_pred, y_pred_length, greedy=True)
        y_pred_label_tensor = y_pred_label_tensor_list[0]
        y_pred_label = keras.backend.get_value(y_pred_label_tensor)
        predict_label[img_path] = (y_pred_label, y_pred_argmax)

        num_images -= 1
    
    return predict_label


# 一次多个图片一起预测
def PredictLabels(model_for_pre, weight_path, test_data_dir, img_size, downsample_factor, batch_size=64):
    img_w, img_h = img_size
    img_path_list = os.listdir(test_data_dir)
    # img_path_list = img_path_list[0: 10]
    num_images = len(img_path_list)
    counter = num_images
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
        y_pred_labels_tensor_list, _ = keras.backend.ctc_decode(y_pred_probMatrix, y_pred_length, greedy=True)
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