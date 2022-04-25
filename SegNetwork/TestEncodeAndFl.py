from EncodeAndFlatten import Network
from CarlaApiAsync import CarlaApi
import cv2
import numpy as np
import copy
# 載入權重
Model = Network().buildModel()
Model.build(input_shape=(75, 100, 64))
Model.load_weights("")


CarlaApi = CarlaApi(img_width=400,img_height=300)
CarlaApi.initial(AutoMode=True)
CarlaApi.wait_for_sim()

try:
    while True:
        ori_frame = CarlaApi.camera_data()['bgr_camera']
        ori_frame_cp = copy.copy(ori_frame)
        ori_frame_cp = ori_frame_cp[np.newaxis,:]

        encode_output = Network().EncodeOutput(ori_frame_cp)
        result = Model.predict(encode_output)

        cv2.imshow('',ori_frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
finally:
    CarlaApi.destroy()
    cv2.destroyAllWindows()
    print('Destroy actor')

