import os
import numpy as np
import cv2
from dicts import num2char_dict, char2num_dict 
class DataGenerator:
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
                # gau_closing_max 化
                # gau = cv2.GaussianBlur(gray_img, (3, 3), 1.5)
                # gau_closing = cv2.morphologyEx(gau, cv2.MORPH_CLOSE, (5, 5))
                # max_gau_closing = np.where(gau_closing < 240, np.zeros(gau_closing.shape), np.ones(gau_closing.shape)*255)

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

def main(dir_path):
    data_train = DataGenerator(dir_path, (128, 32), 4, 32, 12)
    data_train_gen = data_train.get_data()
    for d in data_train_gen:
        print(d)
        break

if __name__ == "__main__":
    main("data/devdata_all")


            


        
