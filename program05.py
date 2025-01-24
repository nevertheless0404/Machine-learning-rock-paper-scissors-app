# 각도에 따라 가위 바위 보 손모양 저장

import cv2
import numpy as np
import pandas as pd
import time
import handTracking as ht

# Landmarks를 받아서 벡터 사이의 각도를 반환
def getAngles(lms):
    # 0번 landmakr 가로, 세로 좌표
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

# 초기화 
recording = False
t_recording = 10
idx = 0 
data_set = []
labels = {0:'가위', 1:'바위', 2:'보'}

my_cap = cv2.VideoCapture(0)                 
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)   
my_detector = ht.HandDetector()                  
while True:
    _, img = my_cap.read()             
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
    imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화
    lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점        

    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  
    cv2.imshow("Image", imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):            
        break
    
    # 'space' 키가 눌려지면 스타트
    if (not recording) and (cv2.waitKey(1) & 0xFF == 32): 
        recording = True
        t_start = time.time()                           
        print('-'*30)
        print(f'"{labels[idx]}" 기록 시작.')
        print(f'남은 시간 = 10초.')

    if recording:
        t_now = time.time()                             
        if (t_now - t_start ) > 1:                      
            t_recording -= 1                           
            print(f'남은 시간 = {t_recording}초.')
            t_start = t_now

            data_set.append(list(getAngles(lms)) + [idx])

            if t_recording == 0:
                print(f'"{labels[idx]}" 기록 끝.')
                print('-'*30)
                idx += 1
                recording = False
                t_recording = 10

                if idx == 3:                            
                    break

# 수집된 데이터 파일로 출력
df = pd.DataFrame(data_set)
df.to_csv('data_train.csv', index=False)

