from SegNetwork import SegNetwork
from tensorflow.keras.callbacks import *
import tensorflow as tf
import numpy as np
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)



# 全部的類別
AllClass = {
    '0': (0, 0, 0),  # 未標記
    '1': (70, 70, 70),  # 建築
    '2': (40, 40, 100),  # 柵欄
    '3': (80, 90, 55),  # 其他
    '4': (60, 20, 220),  # 行人
    '5': (153, 153, 153),  # 桿
    '6': (50, 234, 157),  # 道路線
    '7': (128, 64, 128),  # 馬路
    '8': (232, 35, 244),  # 人行道
    '9': (35, 142, 107),  # 植披
    '10': (142, 0, 0),  # 汽車
    '11': (156, 102, 102),  # 牆
    '12': (0, 220, 220),  # 交通號誌
    '13': (180, 130, 70),  # 天空
    '14': (81, 0, 81),  # 地面
    '15': (100, 100, 150),  # 橋
    '16': (140, 150, 230),  # 鐵路
    '17': (180, 165, 180),  # 護欄
    '18': (30, 170, 250),  # 紅綠燈
    '19': (160, 190, 110),  # 靜止的物理
    '20': (50, 120, 170),  # 動態的
    '21': (150, 60, 45),  # 水
    '22': (100, 170, 145)  # 地形
}
# 感興趣的類別
InterestClass = {
    '0': (0, 0, 0),  # 未標記
    '1': (60, 20, 220),  # 行人
    '2': (50, 234, 157),  # 道路線
    '3': (128, 64, 128),  # 馬路
    '4': (232, 35, 244),  # 人行道
    '5': (142, 0, 0),  # 汽車
    '6': (30, 170, 250)  # 紅綠燈
}
# 總共有幾類
CLASS_NUM = len(InterestClass.items())

# 讀取訓練資料
ori_img_path = './ori'
ori_img_name = os.listdir(ori_img_path)

seg_img_path = './seg'
seg_img_name = os.listdir(seg_img_path)

# 訓練集與驗證集比例
RATIO = 0.9
NUM_TRAINS = int(len(ori_img_name) * RATIO)

# 訓練批次
BATCH_SIZE = 16

# 學習率
LR = 0.002

# 生成器
def generate(data,batch_size=5):
    i = 0
    n = len(data)

    while True:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            # 圖片名稱
            img_name = data[i]

            # X_train圖片處理
            ori_img = cv2.imread(ori_img_path + '/' + img_name)
            X_train.append(ori_img/255)

            # Y_train圖片預處理
            seg_img = cv2.imread(seg_img_path + '/' + img_name)
            seg_img_height = seg_img.shape[0]
            seg_img_width = seg_img.shape[1]
            label_seg = np.zeros((seg_img_height, seg_img_width, CLASS_NUM))

            for index, obj_bgr in enumerate(AllClass.values()):
                b = obj_bgr[0]
                g = obj_bgr[1]
                r = obj_bgr[2]
                matrix = np.where((seg_img[:, :, 0] == b) & (seg_img[:, :, 1] == g) & (seg_img[:, :, 2] == r),
                                   np.ones((seg_img_height, seg_img_width)), np.zeros((seg_img_height, seg_img_width)))

                if obj_bgr in InterestClass.values() and index != 0:
                    channel = list(InterestClass.values()).index(obj_bgr)
                    label_seg[:, : ,channel] = matrix
                else:
                    label_seg[:, :, 0] += matrix

            Y_train.append(label_seg)
            i = (i + 1) % n

        yield (np.array(X_train), np.array(Y_train))



learning_rate_reduction = ReduceLROnPlateau(
                                            monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
checkpoint_period = ModelCheckpoint(
                                    filepath='./weight/ep{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}.h5',
                                    monitor='val_acc',
                                    save_weights_only=True,
                                    save_best_only=True,
                                    period=3
                                )
early_stopping = EarlyStopping(
                                monitor='val_acc',
                                min_delta=0,
                                patience=5,
                                verbose=2
                            )


SegModel = SegNetwork(cls_num=CLASS_NUM,LR=LR)
SegModel = SegModel.getModel()

SegModel.fit(generate(ori_img_name[:NUM_TRAINS],batch_size=BATCH_SIZE),
            steps_per_epoch=max(1, NUM_TRAINS // BATCH_SIZE),
            validation_data=generate(ori_img_name[NUM_TRAINS:], batch_size=BATCH_SIZE),
            epochs=100,
            validation_steps=max(1, len(ori_img_name[NUM_TRAINS:]) // BATCH_SIZE),
            initial_epoch=0,
            callbacks=[learning_rate_reduction,checkpoint_period,early_stopping])