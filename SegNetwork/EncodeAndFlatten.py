from SegEncodeNetwork import EncodeNetwork
from SegNetwork import SegNetwork
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os


class Network(Model):
    def __init__(self):
        super().__init__()
        # 原本的SegNetwork網路的權重位置
        self.SegNetworkWeights = "./weights/ep015-loss0.017-val_acc0.945.h5"
        self.EncodeNetwork = EncodeNetwork()
        self.SetEncodeNetworkWeights()

        self.fl_la = Flatten()
        self.h1 = Dense(32,activation='relu')
        self.h2 = Dense(16,activation='relu')

        self.TL = Dense(1,activation='relu')
        # self.yaw_angel = Dense(1,activation=None)
        # self.la_diff = Dense(1,activation=None)
        self.Intersection = Dense(1,activation=None)

    def SetEncodeNetworkWeights(self):
        self.EncodeNetwork.build(input_shape=(1, 300, 400, 3))

        # 若沒有分割好的EncodeWork的權重，會自動生成一個
        if os.path.isfile('./weights/EncodeNetWorkWeights.h5'):
            self.EncodeNetwork.load_weights('./weights/EncodeNetWorkWeights.h5')
        else:
            SegModel = SegNetwork()
            SegModel = SegModel.loadWeights(self.SegNetworkWeights)
            EncodeNetwork = SegModel.layers.pop(0)

            self.EncodeNetwork.set_weights(EncodeNetwork.get_weights())
            self.EncodeNetwork.save_weights('./weights/EncodeNetWorkWeights.h5')

    def call(self,inputs):
        output = self.EncodeNetwork(inputs)
        output = self.fl_la(output)
        output = self.h1(output)
        output = self.h2(output)

        TL_output = self.TL(output)
        # yaw_angle_output = self.yaw_angel(output)
        # la_diff_output = self.la_diff(output)
        Intersection_output = self.Intersection(output)

        return TL_output, Intersection_output
