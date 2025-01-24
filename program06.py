# 가위, 바위, 보 인식 머신러닝 모형 테스트 

import cv2
import numpy as np
import pandas as pd
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

# 모델 생성 
df = pd.read_csv('data_train.csv')
X = df.drop(columns=['19']).values.astype('float32')
Y = df[['19']].values.astype('float32') 

knn = cv2.ml.KNearest_create()
knn.train(X, cv2.ml.ROW_SAMPLE, Y)

labels = {-1: '---', 0:'Kawi', 1:'Bawi', 2:'Bo'}


# 문자열 삽입 
def insertString(img, text):
    cv2.putText(img=img, text=text,    
                org=(10,70),                            
                fontFace=cv2.FONT_HERSHEY_PLAIN,         
                fontScale=3,                            
                color=(0,0,255),                        
                thickness=3
                )                             

my_cap = cv2.VideoCapture(0)                 
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)   
my_detector = ht.HandDetector()                  
while True:
    _, img = my_cap.read()             
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
    imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화
    lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점 

    if lms:
        angles = getAngles(lms)
        angles = angles[np.newaxis, :]
        pred = knn.findNearest(angles.astype('float32'), 3)
        # print(labels[int(pred[0])])
        insertString(imgRGB, labels[int(pred[0])])
    else:
        insertString(imgRGB, labels[-1])

    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  
    cv2.imshow("Image", imgBGR)

    if cv2.waitKey(1) & 0xFF == ord('q'):            
        break
    

