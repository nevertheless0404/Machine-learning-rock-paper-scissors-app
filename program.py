# openCV 라이브러리 webCam 사용 

import cv2

# 첫번째 카메라
my_cap = cv2.VideoCapture(0)
# 카메라 입력, 높이
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

while True:
    _, img = my_cap.read()
    
    cv2.imshow("Image", img)
    # 'q' 키를 누르면 나감
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break