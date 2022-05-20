from EncodeAndFlatten import Network
from CarlaApiAsync import CarlaApi
import cv2
import numpy as np


# 載入權重
EncodeAndFlatten = Network(now_path='.').buildModel()
EncodeAndFlatten.model.build(input_shape=(75, 100, 64))
EncodeAndFlatten.load_weights()


CarlaApi = CarlaApi(img_width=400,img_height=300)
CarlaApi.initial(AutoMode=True)
CarlaApi.wait_for_sim()

try:
    while True:
        ori_frame = CarlaApi.camera_data()['front_bgr_camera']
        pre_need_slow, pre_tl, pre_tl_dis= EncodeAndFlatten.predict(ori_frame)

        pre_need_slow = np.squeeze(pre_need_slow)
        pre_tl = np.squeeze(pre_tl)
        pre_tl_dis = np.squeeze(pre_tl_dis)

        NeedSlow = 'True' if pre_need_slow > 0.8 else 'False'

        if np.argmax(pre_tl) == 0:
            Pre_TL = 'Green'
        elif np.argmax(pre_tl) == 1:
            Pre_TL = 'Red'
        else:
            Pre_TL = 'None'

        if np.argmax(pre_tl_dis) == 0:
            Pre_TL_dis = 'Close'
        elif np.argmax(pre_tl_dis) == 1:
            Pre_TL_dis = 'Medium'
        elif np.argmax(pre_tl_dis) == 2:
            Pre_TL_dis = 'Far'
        else:
            Pre_TL_dis = 'None'

        print('NeedSlow:',NeedSlow,'TL:',Pre_TL,'TL_dis:',Pre_TL_dis)
        cv2.imshow('',ori_frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
finally:
    CarlaApi.destroy()
    cv2.destroyAllWindows()
    print('Destroy actor')

