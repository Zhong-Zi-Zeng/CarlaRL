from SegNetwork import SegNetwork
from CarlaApiAsync import CarlaApi
import cv2

# 載入權重
SegModel = SegNetwork(cls_num=6)
SegModel.loadWeights("./ep015-loss0.013-val_acc0.976.h5")

CarlaApi = CarlaApi(img_width=400,img_height=300)
CarlaApi.initial(AutoMode=True)
CarlaApi.wait_for_sim()

try:
    while True:
        ori_frame = CarlaApi.camera_data()['bgr_camera']
        output = SegModel.predict(ori_frame)
        seg_frame = SegModel.parseOutput(output)

        cv2.imshow('',seg_frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
finally:
    CarlaApi.destroy()
    cv2.destroyAllWindows()
    print('Destroy actor')

