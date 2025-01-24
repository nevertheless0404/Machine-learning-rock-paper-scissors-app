import cv2
import handTracking as ht

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
        print(lms)

    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  
    cv2.imshow("Image", imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):            
        break