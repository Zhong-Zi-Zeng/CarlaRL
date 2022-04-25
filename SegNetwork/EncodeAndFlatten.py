from SegEncodeNetwork import EncodeNetwork
from SegNetwork import SegNetwork
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os


class Network:
    def __init__(self):
        # 原本的SegNetwork網路的權重位置
        self.SegNetworkWeights = "./weights/ep015-loss0.017-val_acc0.945.h5"
        self.EncodeNetwork = EncodeNetwork()
        self.SetEncodeNetworkWeights()

        self.model = None

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

    def EncodeOutput(self,img):
        return self.EncodeNetwork.predict(img/255)[0]

    def buildModel(self):
        inputs = Input(shape=(75, 100, 64))
        output = GlobalMaxPooling2D()(inputs)

        output_1 = Dense(1024, activation='relu')(output)
        output_1 = Dense(512, activation='relu')(output_1)

        output_2 = Dense(1024, activation='relu')(output)
        output_2 = Dense(512, activation='relu')(output_2)

        TL = Dense(1, activation='sigmoid', name='TL')(output_1)
        Junction = Dense(1, activation='sigmoid', name='Junction')(output_2)
        self.model = Model(inputs=inputs, outputs=[TL, Junction])

        return self

