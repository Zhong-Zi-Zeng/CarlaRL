import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,BatchNormalization
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions):
        super().__init__()

        # self.la1 = BatchNormalization()
        # self.la2 = Conv2D(8, (3, 3), padding='same', activation='relu')
        # self.la3 = Conv2D(8, (3, 3), padding='same', activation='relu')
        # self.la4 = MaxPooling2D((2,2))

        self.la5 = BatchNormalization()
        self.la6 = Conv2D(16, (3, 3), padding='same', activation='relu')
        self.la7 = Conv2D(16, (3, 3), padding='same', activation='relu')
        self.la8 = MaxPooling2D((2,2))

        self.la9 = BatchNormalization()
        self.la10 = Conv2D(32, (3, 3), padding='same', activation='relu')
        self.la11 = Conv2D(32, (3, 3), padding='same', activation='relu')
        self.la12 = MaxPooling2D((2,2))

        self.la13 = BatchNormalization()
        self.la14 = Flatten()
        self.la15_1 = Dense(128, activation='tanh')
        self.la16_1 = Dense(64, activation='tanh')
        self.la15_2 = Dense(128, activation='relu')
        self.la16_2 = Dense(64, activation='relu')

        self.pi = Dense(n_actions, activation='softmax')
        self.v = Dense(1, activation=None)

    def call(self, state):
        # output = self.la1(state)
        # output = self.la2(output)
        # output = self.la3(output)
        # output = self.la4(output)
        output = self.la5(state)
        output = self.la6(output)
        output = self.la7(output)
        output = self.la8(output)
        output = self.la9(output)
        output = self.la10(output)
        output = self.la11(output)
        output = self.la12(output)
        output = self.la13(output)
        output = self.la14(output)

        output_1 = self.la15_1(output)
        output_1 = self.la16_1(output_1)

        output_2 = self.la15_1(output)
        output_2 = self.la16_1(output_2)

        pi = self.pi(output_1)
        v = self.v(output_2)

        return pi ,v