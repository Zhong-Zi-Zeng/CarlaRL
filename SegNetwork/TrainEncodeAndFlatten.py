from EncodeAndFlatten import Network
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import os
import cv2
import linecache
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


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
LR = 0.001

# 生成模型
EncodeAndFlattenNetwork = Network(now_path='.').buildModel()
EncodeAndFlattenNetwork.model.summary()

losses = {'TL': 'binary_crossentropy',
          'Junction': 'binary_crossentropy'}

EncodeAndFlattenNetwork.model.compile(optimizer=RMSprop(learning_rate=LR), loss=losses, metrics=['acc'])


def generate(data,batch_size=5):
    i = 0
    n = len(data)
    while True:
        X_train = []
        TL_label = []
        Injunction_label = []
        for _ in range(batch_size):
            # 圖片名稱
            img_name = data[i]
            # print(img_name)
            # X_train圖片處理
            ori_img = cv2.imread(x_train_path + '/' + img_name)
            ori_img = ori_img[np.newaxis,:]
            encode_output = EncodeAndFlattenNetwork.EncodeOutput(ori_img)
            X_train.append(encode_output)

            # Label 處理
            Junction_label, Tl_label = readSpeficyRow(img_name.strip('.png'))
            TL_label.append(Tl_label)
            Injunction_label.append(Junction_label)
            i = (i + 1) % n

        yield (np.array(X_train), [np.array(TL_label), np.array(Injunction_label)])


def readSpeficyRow(row):
    row = int(row) / 5 + 1
    str = linecache.getline('label.txt', int(row)).rstrip().split(' ')
    Junction_label = int(str[1])
    Tl_label = int(str[2])

    return Junction_label, Tl_label


learning_rate_reduction = ReduceLROnPlateau(
                                            monitor='val_TL_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
checkpoint_period = ModelCheckpoint(
                                    filepath='./val_TL_acc{val_TL_acc:.3f}-val_Junction_acc{val_Junction_acc:.3f}.h5',
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


EncodeAndFlattenNetwork.model.fit(generate(x_train_name[:NUM_TRAINS],batch_size=BATCH_SIZE),
            steps_per_epoch=max(1, NUM_TRAINS // BATCH_SIZE),
            validation_data=generate(x_train_name[NUM_TRAINS:], batch_size=BATCH_SIZE),
            epochs=30,
            validation_steps=max(1, len(x_train_name[NUM_TRAINS:]) // BATCH_SIZE),
            initial_epoch=0,
            callbacks=[learning_rate_reduction,checkpoint_period,early_stopping])