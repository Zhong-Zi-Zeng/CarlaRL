from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
import tensorflow as tf
import cv2
import linecache
import os
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


class Network(Model):
    def __init__(self):
        super().__init__()
        self.filters = [256,128,64,32]
        self.strides = [4,2,1]
        self.hidden_dense = [256,128]

        self.seq_model = Sequential()

        for filter, stride in zip(self.filters, self.strides):
            self.seq_model.add(Conv2D(filter,(3,3),strides=(stride,stride),padding='same',activation='relu'))
            self.seq_model.add(BatchNormalization())
            self.seq_model.add(MaxPooling2D(pool_size=(2,2)))

        self.seq_model.add(GlobalAveragePooling2D())

        for hidden in self.hidden_dense:
            self.seq_model.add(Dense(hidden,activation='relu',use_bias=False))

        self.need_slow_dense = Dense(2, activation='softmax', name='need_slow')
        self.TL_dense = Dense(3, activation='softmax', name='TL')
        self.TL_dis_dense = Dense(4, activation='softmax', name='TL_dis')

    def call(self,inputs):
        output = self.seq_model(inputs)

        # need_slow = self.need_slow_dense(output)
        # TL = self.TL_dense(output)
        # TL_dis_dense = self.TL_dis_dense(output)

        return output

def build_model():
    inputs = Input(shape=(300, 400, 3))
    output = Network()(inputs)

    output_1 = Dense(1024, activation='relu')(output)
    output_1 = Dense(512, activation='relu')(output_1)

    output_2 = Dense(1024, activation='relu')(output)
    output_2 = Dense(512, activation='relu')(output_2)

    output_3 = Dense(1024, activation='relu')(output)
    output_3 = Dense(512, activation='relu')(output_3)

    need_slow = Dense(2, activation='softmax', name='need_slow')(output_1)
    TL = Dense(3, activation='softmax', name='TL')(output_2)
    TL_dis = Dense(4, activation='softmax', name='TL_dis')(output_3)

    model = Model(inputs=inputs, outputs=[need_slow, TL, TL_dis])

    return model


# 載入訓練數據
x_train_path = './data'
x_train_name = os.listdir(x_train_path)
np.random.shuffle(x_train_name)

# 訓練集與驗證集比例
RATIO = 0.7
NUM_TRAINS = int(len(x_train_name) * RATIO)

# 訓練批次
BATCH_SIZE = 64

# 學習率
LEARNING_RARE = 0.01

model = build_model()
opt = Adam(learning_rate=LEARNING_RARE)
losses = {
    'TL': 'categorical_crossentropy',
    'TL_dis': 'categorical_crossentropy',
    'need_slow': 'categorical_crossentropy'
}
model.build(input_shape=(1,300,400,3))
model.compile(opt,loss=losses,metrics=['acc'])
model.summary()


def generate(data,batch_size=5):
    i = 0
    n = len(data)
    while True:
        X_train = []
        TL_label_list = []
        TL_dis_label_list = []
        need_slow_label_list = []
        for _ in range(batch_size):
            # 圖片名稱
            img_name = data[i]
            # X_train圖片處理
            ori_img = cv2.imread(x_train_path + '/' + img_name)
            X_train.append(ori_img)

            # Label 處理
            need_slow_label, Tl_label, TL_dis_label = readSpeficyRow(img_name.strip('.png'))
            need_slow_label_list.append(need_slow_label)
            TL_label_list.append(Tl_label)
            TL_dis_label_list.append(TL_dis_label)
            i = (i + 1) % n

        yield (np.array(X_train), [np.array(need_slow_label_list), np.array(TL_label_list), np.array(TL_dis_label_list)])


def readSpeficyRow(row):
    row = int(row) + 1
    str = linecache.getline('label.txt', int(row)).rstrip().split(' ')
    need_slow_label = list(map(int,str[1:3]))
    Tl_label = list(map(int,str[3:6]))
    TL_dis_label = list(map(int,str[6:]))
    return need_slow_label, Tl_label, TL_dis_label



learning_rate_reduction = ReduceLROnPlateau(
                                            monitor='val_TL_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
checkpoint_period = ModelCheckpoint(
                                    filepath='./val_TL_acc{val_TL_acc:.3f}-val_need_slow_acc{val_need_slow_acc:.3f}.h5',
                                    monitor='val_TL_acc',
                                    save_weights_only=True,
                                    save_best_only=True,
                                    period=3
                                )
early_stopping = EarlyStopping(
                                monitor='val_TL_acc',
                                min_delta=0,
                                patience=5,
                                verbose=2
                            )

model.fit(generate(x_train_name[:NUM_TRAINS],batch_size=BATCH_SIZE),
            steps_per_epoch=max(1, NUM_TRAINS // BATCH_SIZE),
            validation_data=generate(x_train_name[NUM_TRAINS:], batch_size=BATCH_SIZE),
            epochs=50,
            validation_steps=max(1, len(x_train_name[NUM_TRAINS:]) // BATCH_SIZE),
            initial_epoch=0,
            callbacks=[learning_rate_reduction,checkpoint_period,early_stopping]
            )