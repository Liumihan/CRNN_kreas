import os
import cv2
import numpy as np
from keras import backend as K
from dicts import char2num_dict, num2char_dict

def ctc_loss_layer(args):
    '''
    输入：args = (y_true, y_pred, pred_length, label_length)
    y_true, y_pred分别是预测的标签和真实的标签
    shape分别是（batch_size，max_label_length)和(batch_size, time_steps, num_categories)
    perd_length, label_length分别是保存了每一个样本所对应的预测标签长度和真实标签长度
    shape分别是（batch_size, 1)和(batch_size, 1)
    输出：
    batch_cost 每一个样本所对应的loss
    shape是（batch_size, 1)
    '''
    y_true, y_pred, pred_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    batch_cost = K.ctc_batch_cost(y_true, y_pred, pred_length, label_length)
    return batch_cost


def fake_ctc_loss(y_true, y_pred):
    '''
    这个函数是为了符合keras comepile的要求入口参数只能有y_true和y_pred
    之后在结合我们的ctc_loss_layer一起工作
    '''
    return y_pred

def check_acc(predict_labels):
    acc = 0
    misclassified = {}
    for gt, pre in predict_labels.items():
        if gt.split("_")[0] == pre:
            acc += 1
        else: 
            misclassified[gt] = pre
    acc /= len(predict_labels)
    return acc, misclassified


def find_the_inner_dot(dir_path):
    img_list = os.listdir(dir_path)
    for img in img_list:
        for i in range(len(img)):
            if img[i] == '.' and img[i+1] != 'j':
                print(img)
                continue


def find_the_max_label_length(dir_path):
    max_label_length = 0
    img_list = os.listdir(dir_path)
    for img in img_list:
        label = img.split('_')[0]
        if len(label) > max_label_length:
            max_label_length = len(label)
    print("max_label_length: ", max_label_length)
    return max_label_length

def closure(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # opening_kernel = (5,5)
    # closing_kernel = (5,5)
    # gau_kernel_size = (5,5)
    # dil_kernel = (5,5)
    sigma = 1.5  
    cv2.imshow("before", img)
    cv2.moveWindow("before", 900, 1200)
    for ga in range(3, 8, 2):
        gau = cv2.GaussianBlur(img, (ga, ga), 1.5)
        for cl in range(3, 6, 1):
            gau_closing = cv2.morphologyEx(gau, cv2.MORPH_CLOSE, (cl, cl))
            max_gau_closing = np.where(gau_closing < 240, np.zeros(gau_closing.shape), np.ones(gau_closing.shape)*255)
            # max_gau_closing = cv2.GaussianBlur(max_gau_closing, (ga, ga), 2.5)
            # max_gau_closing = cv2.morphologyEx(max_gau_closing, cv2.MORPH_CLOSE, (cl, cl))
            cv2.imshow("max_gau_closing_ga{}_cl{}".format(ga, cl), max_gau_closing)
            cv2.moveWindow("max_gau_closing_ga{}_cl{}".format(ga, cl), (ga-3)*200, (cl-3)*200)
    cv2.waitKey(0) 
    return 0

def generate_trainfile(data_dir_path, save_file_path):
    '''
    生成类似 300w+的那个数据集类似的txt文件
    保存格式： 图片文件名.jpg<空格>对应标签
    '''
    img_path_list = os.listdir(data_dir_path)
    target_file = open(save_file_path, "a")
    for img_filename in img_path_list:
        gt = img_filename.split("_")[0]
        gt_nums = [char2num_dict[ch] for ch in gt] # 转换成对应的数字标签 
        target_file.write(img_filename)
        for num in gt_nums:
            target_file.write(" " + str(num)) # 在每两个数字之间增加一个空格
        target_file.write("\n") # 末尾换行
    target_file.close()
    return 0

def main():
    generate_trainfile("../data/numbers_val_croped", "../data/data_txt/val.txt")

    return 0

if __name__ == "__main__":
    # find_the_inner_dot("../data/numbers_training")
    # find_the_max_label_length("../data/numbers_training")
    # img = cv2.imread("../data/numbers_training_croped/30065864_20071.jpg")
    # closure(img)
    # a = find_the_inner_dot("../data/numbers_val_croped")
    main()