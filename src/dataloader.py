#-*- coding-utf8 -*-
import cv2
import os
import numpy as np

char2num_dict = {'0': 0, '1': 1, '2':2, '3': 3, '4': 4, 
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '_': 10}
num2char_dict = {value : key for key, value in char2num_dict.items()}


def load_data(fileDirPath):
    '''
    输入：str 
    图片的文件夹路径

    返回：np.array, np.array
    data 和 label 都是array，data中的array保存的是
    转成了灰度图的图片（类型是array，shape=(H, W, C)）
    图片被resize成了（32, 128, 1),并且都除以了255
    labels中保存的是对应图片的标签，一一对应，即data[i]的标签是label[i] 
    label[i] 的是一个array，长度与字符串长度对应每一个元素是对应位置的字符在char2num中对应的数字。
    '''
    files = os.listdir(fileDirPath)
    
    data, labels = [], []
    labels_length = np.zeros((len(files), 1), dtype=np.int64)
    #pred_label_length我直接设置成了固定值应为我的网络最后输出的就是32个timestep
    pred_labels_length = np.full((len(files), 1), 32, dtype=np.int64)
    picture_width = 128
    max_labels_length = 12

    for i, f in enumerate(files):
        # 1 生成data
        img = cv2.imread(os.path.join(fileDirPath, f))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray_img.shape
        gray_img = cv2.resize(gray_img, (picture_width, 32))
         
        data.append(gray_img)
        # cv2.imshow('gray', gray_img)
        # cv2.waitKey(0)
        # 2 生成对应的label
        str_label = f.split('_')[0]
        labels_length[i][0] = len(str_label) # 记录真实每个真实label的length用来计算ctc_loss
        num_label = [char2num_dict[ch] for ch in str_label] # 已经转换成数字，但是还没有同样长度的label
        for n in range(max_labels_length - len(str_label)):
            num_label.append(char2num_dict['_'])
        labels.append(num_label)
    data = np.array(data) / 255.0 * 2.0 -1.0 # 待会儿改回255
    data = np.expand_dims(data, axis=-1)
    labels = np.array(labels)
    labels_length = np.array(labels_length)
    return labels, data, pred_labels_length, labels_length


def main():
    labels, data, pred_labels_length, labels_length = load_data("data/devdata")


    return 0

if __name__ == "__main__":
    main()