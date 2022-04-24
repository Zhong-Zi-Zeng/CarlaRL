from SegNetwork import SegNetwork
import cv2
import os


# 測試圖片位置
test_img_path = './test_img'

SegModel = SegNetwork(cls_num=6)
SegModel.loadWeights("./weights/ep015-loss0.017-val_acc0.945.h5")

for img_name in os.listdir(test_img_path):
    test_img = cv2.imread(test_img_path + '/' + img_name)
    output = SegModel.predict(test_img)
    show_img = SegModel.parseOutput(output)

    cv2.imshow('',show_img)
    cv2.waitKey(0)



