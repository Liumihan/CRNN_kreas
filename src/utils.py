import os
import cv2
import numpy as np
from keras import backend as K
from dicts import char2num_dict, num2char_dict
import random

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

def check_acc_by_filename(predict_labels):
    acc = 0
    misclassified = {}
    for gt, pre in predict_labels.items():
        if gt.split("_")[0] == pre:
            acc += 1
        else: 
            misclassified[gt] = pre
    acc /= len(predict_labels)
    return acc, misclassified

def check_acc(predict_labels, txt_file_path):
    acc = 0
    misclassified = {}
    data_txt = open(txt_file_path, "r")
    data_txt_list = data_txt.readlines()
    ground_truthes = {}
    for line in data_txt_list:
        img_name = line.split(" ")[0]
        true_label = ""
        for ch in line.split("\n")[0].split(" ")[1:]:
            true_label += num2char_dict[int(ch)]
        ground_truthes[img_name] = true_label

    for img_name, pre in predict_labels.items():
        if pre == ground_truthes[img_name]:
            acc += 1
        else: 
            misclassified[img_name] = pre
    acc /= len(predict_labels)
    data_txt.close()
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
    max_idx = 0
    counter = 1
    for i, img in enumerate(img_list):
        label = img.split('_')[0]
        if len(label) == max_label_length:
            counter += 1
        if len(label) > max_label_length:
            counter = 1
            max_label_length = len(label)
            max_idx = i
    print("max_label_length: ", max_label_length)
    print("max_label_img: ", img_list[max_idx])
    print("counter: ", counter)
    return max_label_length

def find_the_max_label_length_txt(txt_file_path):
    max_label_length = 0
    data_txt = open(txt_file_path, "r")
    data_txt_list = data_txt.readlines()
    counter = 1
    for line in data_txt_list:
        true_label_length = len(line.split("\n")[0].split(" ")[1:])
        if true_label_length == max_label_length:
            counter += 1
        if true_label_length > max_label_length:
            counter = 1
            max_label_length = true_label_length
    data_txt.close()
    print("max_label_length: ", max_label_length)
    print("counter: ", counter)
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

def generate_txt_file(data_dir_path, save_file_path):
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

def generate_train_test_file(txt_file_path, file_save_dir):
    data_txt = open(txt_file_path, "r")
    test_txt = open(os.path.join(file_save_dir, "test_data.txt"), "a")
    train_txt = open(os.path.join(file_save_dir, "train_data.txt"), "a")
    data_txt_list = data_txt.readlines()
    # 随机选择0.1 的部分作为测试集合
    test_list = random.sample(data_txt_list, int(len(data_txt_list)*0.1))
    # 剩余部分作为 训练集合
    for t in test_list:
        data_txt_list.remove(t)
        test_txt.write(t)
    train_txt_list = data_txt_list

    # 将内容写入txt文件中
    for tr in train_txt_list:
        train_txt.write(tr)

    data_txt.close()   
    test_txt.close()
    train_txt.close()
    return 0

def generate_dict(data_dir_path):
    chars = set()
    img_list = os.listdir(data_dir_path)
    for img in img_list:
        label = img.split('_')[0]
        for ch in label:
            chars.add(ch)
    CharToNumDict = {}
    counter = 0
    for ch in chars:
        CharToNumDict[ch] = counter
        counter += 1
    
    return CharToNumDict

def extract_300w(imgdir_path, savedir_path, txt_file_path, save_txt_path):
    data_txt = open(txt_file_path, "r") # 从中读取文件名 以及对应的标签
    save_txt = open(save_txt_path, "a") # 保存到这个文件夹里面去
    data_txt_list = data_txt.readlines()

    data_txt_list = data_txt_list[0:10000] # 只取前1w个
    counter = 10000
    # 取出文件名并且保存到另一个文件夹
    for line in data_txt_list:
        print("{} files left".format(counter))
        img_filename = line.split(" ")[0]
        img_filepath = os.path.join(imgdir_path, img_filename)
        img = cv2.imread(img_filepath) # 从原始文件夹中读取出图片
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(savedir_path, img_filename), img)
        save_txt.write(line)
        counter -= 1
    data_txt.close()
    save_txt.close()
    return 0


def main():
    # generate_trainfile("../data/numbers_training_croped", "../data/data_txt/train.txt")
    # val_txt = open("../data/data_txt/val.txt", "r")
    # val_txt_list =  val_txt.readlines()[:32]
    # val_img_names = [line.split(" ")[0] for line in val_txt_list]
    # val_img_labels_chars = [line.split("\n")[0].split(" ")[1:] for line in val_txt_list ] # 第一个split是为了去掉末尾的"\n" 第二个是为了去掉空格 现在里面存的是字符数组
    # val_img_labels_nums = [] # 现在将他转换成数字数组
    # val_txt.close()

    max_length = find_the_max_label_length_txt("../data/part_300w/txt/all.txt")

    # m_dict = generate_dict("../data/all_data")
    # max_l = find_the_max_label_length("../data/numbers_croped")
    # generate_txt_file("../data/日期_croped", "../data/data_txt/all_except_long/all_data.txt")
    # a = generate_train_test_file("../data/data_txt/all_except_long/all_data.txt", "../data/data_txt/all_except_long/")

    # extract_300w("../data/images_300W", "../data/part_300w/img", "../data/img_300w_txt/train.txt", "../data/part_300w/txt/all.txt")
    # generate_train_test_file("../data/part_300w/txt/all.txt", "../data/part_300w/txt/")
    return 0

if __name__ == "__main__":
    # find_the_inner_dot("../data/numbers_training")

    # img = cv2.imread("../data/numbers_training_croped/30065864_20071.jpg")
    # closure(img)
    # a = find_the_inner_dot("../data/numbers_val_croped")
    main()