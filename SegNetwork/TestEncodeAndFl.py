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
        tl, junction = EncodeAndFlatten.predict(ori_frame)
        tl = np.squeeze(tl)
        junction = np.squeeze(junction)
        tl = 1 if tl > 0.8 else 0
        junction = 1 if junction > 0.8 else 0
        print('TL:',tl,'Slow:',junction)


        cv2.imshow('',ori_frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
finally:
    CarlaApi.destroy()
    cv2.destroyAllWindows()
    print('Destroy actor')

