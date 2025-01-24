# 각도 추출

import cv2
import numpy as np
import handTracking as ht

# Landmarks를 받아서 벡터 사이의 각도를 반환
def getAngles(lms):
    base = lms[0][1:]
    lms = np.array( [  (x,y) for id, x, y in lms ])
    vectors = lms[1:] - np.array([base]*20)
    # 마디마디를 연결해서 벡터를 만듬 
    # 축이 하나 밖에 없음
    norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    # 축을 2개가 되도록 구성
    vectors = vectors/norms
    # 길이가 1인 벡터로 정규화 
    cos = np.einsum( 'ij,ij->i', vectors[:-1], vectors[1:] )
    # Degree 변환
    angles = np.arccos(cos)*180/np.pi
    return angles

my_cap = cv2.VideoCapture(0)                 
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)   
my_detector = ht.HandDetector()                  
while True:
    _, img = my_cap.read()               
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
    imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화
    lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점        

    # 검출이 성공한 경우만 출력.
    if lms:
        print(getAngles(lms))

    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  
    cv2.imshow("Image", imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):            
        break