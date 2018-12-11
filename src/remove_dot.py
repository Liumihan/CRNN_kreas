import os
import cv2
import numpy as np
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
    # # erosion = cv2.erode(img,kernel,iterations = 1)
    # cv2.imshow("before", img)
    
    # gau = cv2.GaussianBlur(img, gau_kernel_size, sigma)
    # dilation = cv2.dilate(img,dil_kernel,iterations = 1)
    # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernel)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, opening_kernel)


    # gau_closing = cv2.morphologyEx(gau, cv2.MORPH_CLOSE, closing_kernel)
    # gau_dilation = cv2.dilate(gau,dil_kernel,iterations = 1)
    # closing_gau = cv2.GaussianBlur(closing, gau_kernel_size, sigma)

    # max_gau_closing = np.where(gau_closing < 200, np.zeros(gau_closing.shape), np.ones(gau_closing.shape)*255)
    
    # cv2.imshow("opening", opening)
    # cv2.imshow("max_gau_closing", max_gau_closing)
    # cv2.imshow("gau_dilation", gau_dilation)
    # cv2.imshow("gau_closing", gau_closing)
    # cv2.imshow("gau", gau)
    # cv2.imshow("dilation", dilation)
    # cv2.imshow("closing", closing)
    # cv2.imshow("closing_gau", closing_gau)

    # cv2.moveWindow("before", 0,0)
    # cv2.moveWindow("gau", 0, 200)
    # cv2.moveWindow("dilation", 0, 400)
    # cv2.moveWindow("closing", 0, 600)
    # cv2.moveWindow("max_gau_closing", 0, 800)
    # cv2.moveWindow("gau_dilation", 400, 0)
    # cv2.moveWindow("gau_closing", 800, 0)
    # cv2.moveWindow("closing_gau", 400, 200)
    # cv2.moveWindow("opening", 400, 400)
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

if __name__ == "__main__":
    # find_the_inner_dot("../data/numbers_training")
    # find_the_max_label_length("../data/numbers_training")
    img = cv2.imread("../data/numbers_training_croped/30065864_20071.jpg")
    closure(img)