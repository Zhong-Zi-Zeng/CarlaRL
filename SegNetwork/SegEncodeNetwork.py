from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


class BlockLayer(Model):
    def __init__(self,filters,max_flag=False):
        super().__init__()
        self.filters = filters
        self.max_flag = max_flag

        self.conv_la = Conv2D(self.filters, (3, 3), strides=(1,1), padding='same', use_bias=False)
        self.bn_la = BatchNormalization()
        self.max_la = MaxPooling2D((2,2))
        self.ac_la = Activation('relu')

    def call(self,inputs):
        output = self.conv_la(inputs)
        output = self.bn_la(output)
        if self.max_flag:
            output = self.max_la(output)

        output = self.ac_la(output)

        return output


class EncodeNetwork(Model):
    """
        Input Shape = (1,300,400,3)
        Output Shape = (75,100,64)
    """
    def __init__(self):
        super().__init__()
        self.filters_list = [8,16,32,64]
        self.layerSequential = Sequential()

        for i,filters in enumerate(self.filters_list):
            for _ in range(2):
                if i == 0:
                    layer = BlockLayer(filters,max_flag=True)
                else:
                    layer = BlockLayer(filters,max_flag=False)
                self.layerSequential.add(layer)

    def call(self, inputs):
        output = self.layerSequential(inputs)

        return output
