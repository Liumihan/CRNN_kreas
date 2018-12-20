
chars_ptr = open("../data/img_300w/txt/m_char_std_5990.txt", "r")
chars = chars_ptr.read().splitlines()
chars[0] = "，" # 好像转换成utf-8后第一个“，”有点问题
chars_ptr.close()

def get_dict(chars):
    counter = 1 # 从 1 开始 舍弃了第一个存储位置
    char2num_dict = {}
    for ch in chars:
        char2num_dict[ch] = counter
        counter += 1
    return char2num_dict

char2num_dict = get_dict(chars)
num2char_dict = {value : key for key, value in char2num_dict.items()}


# 原始的版本
# chars = "0123456789年月日￥.-_"
# char2num_dict = {'0': 0, '1': 1, '2':2, '3': 3, 
#                 '4': 4, '5': 5, '6': 6, '7': 7, 
#                 '8': 8, '9': 9, '年': 10, '月': 11, 
#                 '日': 12, '￥': 13, '.': 14, '-': 15, '_': 16}
# num2char_dict = {value : key for key, value in char2num_dict.items()}

# 只有数字的版本
# char2num_dict = {'0': 0, '1': 1, '2':2, '3': 3, 
#                 '4': 4, '5': 5, '6': 6, '7': 7, 
#                 '8': 8, '9': 9, '_': 10}
# num2char_dict = {value : key for key, value in char2num_dict.items()}