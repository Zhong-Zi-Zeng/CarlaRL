from EncodeAndFlatten import Network
from CarlaApiAsync import CarlaApi
import cv2


# 載入權重
EncodeAndFlatten = Network().buildModel()
EncodeAndFlatten.model.build(input_shape=(75, 100, 64))
EncodeAndFlatten.load_weights("")


CarlaApi = CarlaApi(img_width=400,img_height=300)
CarlaApi.initial(AutoMode=True)
CarlaApi.wait_for_sim()

try:
    while True:
        ori_frame = CarlaApi.camera_data()['bgr_camera']
        result = EncodeAndFlatten.predict(ori_frame)
        print(result)
        cv2.imshow('',ori_frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
finally:
    CarlaApi.destroy()
    cv2.destroyAllWindows()
    print('Destroy actor')

