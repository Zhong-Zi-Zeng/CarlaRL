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
BATCH_SIZE = 128

# 學習率
LR = 0.001

# 生成模型
EncodeAndFlattenNetwork = Network(now_path='.').buildModel()
EncodeAndFlattenNetwork.model.summary()

losses = {'TL': 'categorical_crossentropy',
          'TL_dis': 'categorical_crossentropy',
          'need_slow': 'categorical_crossentropy'}

EncodeAndFlattenNetwork.model.compile(optimizer=Adam(learning_rate=LR), loss=losses, metrics=['acc'])


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
            ori_img = ori_img[np.newaxis,:]
            encode_output = EncodeAndFlattenNetwork.EncodeOutput(ori_img)
            X_train.append(encode_output)

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


EncodeAndFlattenNetwork.model.fit(generate(x_train_name[:NUM_TRAINS],batch_size=BATCH_SIZE),
            steps_per_epoch=max(1, NUM_TRAINS // BATCH_SIZE),
            validation_data=generate(x_train_name[NUM_TRAINS:], batch_size=BATCH_SIZE),
            epochs=50,
            validation_steps=max(1, len(x_train_name[NUM_TRAINS:]) // BATCH_SIZE),
            initial_epoch=0,
            callbacks=[learning_rate_reduction,checkpoint_period,early_stopping])