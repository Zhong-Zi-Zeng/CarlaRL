from SegEncodeNetwork import EncodeNetwork
from SegNetwork import SegNetwork
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os
import numpy as np


class Network:
    def __init__(self,now_path=''):
        self.now_path = now_path
        # 原本的SegNetwork網路的權重位置
        self.SegNetworkWeights = self.now_path + '/' +"weights/ep015-loss0.017-val_acc0.945.h5"
        self.EncodeNetwork = EncodeNetwork()
        self.SetEncodeNetworkWeights()

        # Flatten的權重位置
        self.FlattenWeights = self.now_path + '/' + 'weights/val_TL_acc0.920-val_Junction_acc0.904.h5'

        self.model = None

    def SetEncodeNetworkWeights(self):
        self.EncodeNetwork.build(input_shape=(1, 300, 400, 3))

        # 若沒有分割好的EncodeWork的權重，會自動生成一個
        if os.path.isfile(self.now_path + '/' + 'weights/EncodeNetWorkWeights.h5'):
            self.EncodeNetwork.load_weights(self.now_path + '/' + 'weights/EncodeNetWorkWeights.h5')
        else:
            SegModel = SegNetwork()
            SegModel = SegModel.loadWeights(self.SegNetworkWeights)
            EncodeNetwork = SegModel.layers.pop(0)

            self.EncodeNetwork.set_weights(EncodeNetwork.get_weights())
            self.EncodeNetwork.save_weights(self.now_path + '/' + 'weights/EncodeNetWorkWeights.h5')

    def EncodeOutput(self,img):
        return self.EncodeNetwork.predict(img/255)[0]

    def buildModel(self):
        inputs = Input(shape=(75, 100, 64))
        output = GlobalMaxPooling2D()(inputs)

        output_1 = Dense(1024, activation='relu')(output)
        output_1 = Dense(512, activation='relu')(output_1)

        output_2 = Dense(1024, activation='relu')(output)
        output_2 = Dense(512, activation='relu')(output_2)

        output_3 = Dense(1024, activation='relu')(output)
        output_3 = Dense(512, activation='relu')(output_3)

        need_slow = Dense(1, activation='sigmoid', name='need_slow')(output_1)
        TL = Dense(3, activation='sigmoid', name='TL')(output_2)
        TL_dis =  Dense(4, activation='sigmoid', name='TL_dis')(output_3)


        self.model = Model(inputs=inputs, outputs=[need_slow, TL, TL_dis])

        return self

    def load_weights(self):
        self.model.load_weights(self.FlattenWeights)

    def predict(self,img):
        img = img[np.newaxis,:]
        encode_output = self.EncodeOutput(img)
        encode_output = encode_output[np.newaxis,:]
        fl_output = self.model.predict(encode_output)

        return fl_output

