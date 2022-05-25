from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

"""
    Input Shape = (1,300,400,512)
    Output Shape = (1,300,400,cls_num)
"""
class DecodeNetwork(Model):
    def __init__(self,cls_num):
        super().__init__()
        self.cls_num = cls_num

        self.h1 = Conv2DTranspose(128,(3,3),strides=(1,1),padding='same',use_bias=False,activation='relu')
        self.b1 = BatchNormalization()

        self.h2 = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False,activation='relu')
        self.b2 = BatchNormalization()

        self.h3 = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False,activation='relu')
        self.b3 = BatchNormalization()

        self.o = Dense(self.cls_num,activation='softmax')

    def call(self, inputs):
        output = self.h1(inputs)
        output = self.b1(output)
        output = self.h2(output)
        output = self.b2(output)
        output = self.h3(output)
        output = self.b3(output)
        output = self.o(output)

        return output


