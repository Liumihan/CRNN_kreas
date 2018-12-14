import os
import numpy as np
import cv2
from dicts import num2char_dict, char2num_dict 
class DataGenerator:
    '''
    这个类的目的是写一个生成器， 节约内存
    文件名包含了数据的gt
    '''
    def __init__(self, img_dirpath, img_size, down_sample_factor, batch_size, max_label_length):
        self.img_dirpath = img_dirpath
        self.img_w ,self.img_h = img_size
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.img_list = np.array(os.listdir(img_dirpath))
        self.each_pred_label_length = int(self.img_w // down_sample_factor)
        self.img_number = len(self.img_list)
        self.char2num_dict = char2num_dict
        self.num2char_dict = num2char_dict
    def get_data(self, is_training=True):
        labels_length = np.zeros((self.batch_size,1))
        pred_labels_length = np.full((self.batch_size, 1), self.each_pred_label_length, dtype=np.float64)
        while True:
            data, labels = [], []
            np.random.shuffle(self.img_list)
            img_to_network = self.img_list[0:self.batch_size]
            for i, img_file in enumerate(img_to_network):
                img = cv2.imread(os.path.join(self.img_dirpath, img_file))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.resize(gray_img, (self.img_w, self.img_h))
                gray_img = gray_img.astype(np.float32)

                data.append(gray_img)
                str_label = img_file.split('_')[0]
                labels_length[i][0] = len(str_label)
                num_label = [self.char2num_dict[ch] for ch in str_label]
                for n in range(self.max_label_length - len(str_label)):
                    num_label.append(self.char2num_dict['_'])
                labels.append(num_label)
            data = np.array(data, dtype=np.float64) / 255.0 * 2 - 1 # 零中心化
            data = np.expand_dims(data, axis=-1)
            labels = np.array(labels, dtype=np.float64)
            inputs={"y_true": labels,  
                    "pic_inputs": data, 
                    "y_pred_length": pred_labels_length,
                    "y_true_length": labels_length}
            outputs={"ctc_loss_output": np.zeros((self.batch_size, 1), dtype=np.float64)}
            if is_training:      
                yield (inputs, outputs)
            else:
                yield (data, pred_labels_length)

class DataGenerator_txt:
    '''
    这个类是从txt文件里面读取图片文件名，以及对应的gt
    '''
    def __init__(self, img_dirpath, txt_file_path, img_size, down_sample_factor, batch_size, max_label_length):
        self.img_dirpath = img_dirpath
        self.txt_file_path = txt_file_path
        self.img_w ,self.img_h = img_size
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.each_pred_label_length = int(self.img_w // down_sample_factor)

        # 从txt文件中获取文件名以及所对应的标签
        data_txt = open(self.txt_file_path, "r")
        data_txt_list = data_txt.readlines()
        self.img_list = [line.split(" ")[0] for line in data_txt_list] # 里面保存了所有的数据的文件名
        self.img_list = np.array(self.img_list) # 将他转换成array
        self.img_labels_chars_list = [line.split("\n")[0].split(" ")[1:] for line in data_txt_list]
        self.img_labels_chars_list = np.array(self.img_labels_chars_list)
        self.img_number = len(self.img_list)
        data_txt.close()
        
        self.char2num_dict = char2num_dict
        self.num2char_dict = num2char_dict
    def get_data(self, is_training=True):


        labels_length = np.zeros((self.batch_size,1))
        pred_labels_length = np.full((self.batch_size, 1), self.each_pred_label_length, dtype=np.float64)
        while True:
            data, labels = [], []
            to_network_idx = np.random.choice(self.img_number, self.batch_size, replace=False)
            img_to_network = self.img_list[to_network_idx]
            correspond_labels = self.img_labels_chars_list[to_network_idx]
            for i, img_file in enumerate(img_to_network):
                img = cv2.imread(os.path.join(self.img_dirpath, img_file))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.resize(gray_img, (self.img_w, self.img_h))
                gray_img = gray_img.astype(np.float32)

                data.append(gray_img)
                str_label = correspond_labels[i]
                labels_length[i][0] = len(str_label)
                num_label = [int(ch) for ch in str_label]
                for n in range(self.max_label_length - len(str_label)):
                    num_label.append(self.char2num_dict['_'])
                labels.append(num_label)
            data = np.array(data, dtype=np.float64) / 255.0 * 2 - 1 # 零中心化
            data = np.expand_dims(data, axis=-1)
            labels = np.array(labels, dtype=np.float64)
            inputs={"y_true": labels,  
                    "pic_inputs": data, 
                    "y_pred_length": pred_labels_length,
                    "y_true_length": labels_length}
            outputs={"ctc_loss_output": np.zeros((self.batch_size, 1), dtype=np.float64)}
            if is_training:      
                yield (inputs, outputs)
            else:
                yield (data, pred_labels_length)


def main(dir_path):
    # data_train = DataGenerator(dir_path, (128, 32), 4, 32, 12)
    txt_file_path = "../data/data_txt/val.txt"
    data_train = DataGenerator_txt(dir_path,txt_file_path, (128, 32), 4, 32, 12)
    data_train_gen = data_train.get_data()
    for d in data_train_gen:
        print(d)
        break

if __name__ == "__main__":
    main("../data/numbers_val_croped")


            


        
