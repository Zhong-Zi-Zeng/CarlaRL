from SegDecodeNetwork import DecodeNetwork
from SegEncodeNetwork import EncodeNetwork
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


class SegNetwork:
    def __init__(self,cls_num=7,LR=0.001):
        self.cls_num = cls_num
        self.seg = Sequential()
        self.seg.add(EncodeNetwork())
        self.seg.add(DecodeNetwork(cls_num=self.cls_num))
        self.seg.build(input_shape=(1,300,400,3))
        self.seg.summary()
        self.seg.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['acc'])

    def getModel(self):
        return self.seg

    def loadWeights(self,h5_file):
        self.seg.load_weights(h5_file)
        return self.seg

    def predict(self,input):
        input = input[np.newaxis,:]
        return self.seg.predict(input/255)[0]

    def parseOutput(self,output):
        # 感興趣的類別del car_data['tl']
        #         del car_data['del']
        InterestClass = {
            '0': (0, 0, 0),  # 未標記
            '1': (60, 20, 220),  # 行人
            '2': (50, 234, 157),  # 道路線
            '3': (128, 64, 128),  # 馬路
            '4': (232, 35, 244),  # 人行道
            '5': (142, 0, 0),  # 汽車
            '6': (30, 170, 250)  # 紅綠燈
        }
        result_img = np.zeros((300,400,3), dtype=np.uint8)
        hot_code = np.argmax(output, axis=2)

        for label in range(self.cls_num):
            matrix = np.where((hot_code[:,:] == label),np.ones((300,400)),np.zeros((300,400)))
            result_img[matrix > 0] = InterestClass[str(label)]

        return result_img