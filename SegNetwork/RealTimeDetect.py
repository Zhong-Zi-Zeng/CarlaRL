from SegNetwork import SegNetwork
from CarlaApiAsync import CarlaApi
import cv2
import numpy as np
import tensorflow as tf


# 載入權重
SegModel = SegNetwork(cls_num=5)
SegModel.loadWeights("./weights/ep063-loss0.011-val_acc0.996.h5")

CarlaApi = CarlaApi(img_width=400,img_height=300)
CarlaApi.initial(AutoMode=True)
CarlaApi.wait_for_sim()

try:
    while True:
        ori_frame = CarlaApi.camera_data()['front_bgr_camera']
        output = SegModel.predict(ori_frame)
        seg_frame = SegModel.parseOutput(output)
        show_frame = np.hstack((seg_frame,ori_frame))

        cv2.imshow('',show_frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
finally:
    CarlaApi.destroy()
    cv2.destroyAllWindows()
    print('Destroy actor')

