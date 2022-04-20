from SegDecodeNetwork import DecodeNetwork
from SegEncodeNetwork import EncodeNetwork
from tensorflow.keras.models import *
import tensorflow as tf
import numpy as np
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

# 測試圖片位置
test_img_path = './test'

# 感興趣的類別
InterestClass = {
    '0': (0, 0, 0),  # 未標記
    '1': (60, 20, 220),  # 行人
    '2': (50, 234, 157),  # 道路線
    '3': (128, 64, 128),  # 馬路
    '4': (232, 35, 244),  # 人行道
    '5': (142, 0, 0),  # 汽車
}

# 總共有幾類
CLASS_NUM = len(InterestClass.items())

EncodeNetwork = EncodeNetwork()
DecodeNetwork = DecodeNetwork(cls_num=CLASS_NUM)
SegModel = Sequential()
SegModel.add(EncodeNetwork)
SegModel.add(DecodeNetwork)
SegModel.load_weights('')


for img_name in os.listdir(test_img_path):
    test_img = cv2.imread(test_img_path + '/' + img_name)
    test_img = test_img[np.newaxis,:]
    output = SegModel.predict(test_img/255)

    output_height = output.shape[0]
    output_width = output.shape[1]
    hot_code = np.argmax(output, axis=2)
    show_img = np.zeros_like(test_img)

    for label in InterestClass.keys():
        matrix = np.where(hot_code[:,:] == label,np.ones((output_height, output_width)),
                              np.zeros((output_height, output_width)))

        show_img[matrix == 1] = InterestClass[str(label)]

    result = np.vstack((show_img,test_img))
    cv2.imshow('',result)
    cv2.waitKey(0)



