from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


class BlockLayer(Model):
    def __init__(self,filters,up_flag=False):
        super().__init__()
        self.filters = filters
        self.up_flag = up_flag

        self.up_la = UpSampling2D((2, 2))
        self.conv_la = Conv2DTranspose(self.filters, (3, 3), strides=(1,1), padding='same', use_bias=False)
        self.bn_la = BatchNormalization()
        self.ac_la = Activation('relu')

    def call(self,inputs):
        output = self.conv_la(inputs)
        output = self.bn_la(output)
        if self.up_flag:
            output = self.up_la(output)
        output = self.ac_la(output)

        return output

"""
    Input Shape = (1,75,100,64)
    Output Shape = (1,300,400,cls)
"""
class DecodeNetwork(Model):
    def __init__(self,cls_num):
        super().__init__()
        self.filters_list = [64,32,16]
        self.cls_num = cls_num
        self.layerSequential = Sequential()

        self.o = Dense(self.cls_num,activation='softmax')

        for i,filters in enumerate(self.filters_list):
            for _ in range(2):
                if i == 0:
                    layer = BlockLayer(filters,up_flag=True)
                else:
                    layer = BlockLayer(filters,up_flag=False)
                self.layerSequential.add(layer)

    def call(self, inputs):
        output = self.layerSequential(inputs)
        output = self.o(output)

        return output


