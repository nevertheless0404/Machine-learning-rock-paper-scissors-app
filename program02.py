# Hand Tracking 최소 코드

import cv2
import mediapipe as mp

my_hands = mp.solutions.hands.Hands(                   
            static_image_mode=False,          # 트래킹 병행의 의미
            max_num_hands=1,                  # 손의 갯수
            min_detection_confidence=0.5,     # 검출 최소 한계
            min_tracking_confidence=0.5       # 트래킹 최소 한계
)      

# 첫번째 카메라
my_cap = cv2.VideoCapture(0)
# 카메라 입력, 높이
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

while True:
    _, img = my_cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = my_hands.process(imgRGB)

    if(res.multi_hand_landmarks):
        # 첫 번째 손만 사용
        a_hand = res.multi_hand_landmarks[0]
        # 연결선을 그어줘야 함
        mp.solutions.drawing_utils.draw_landmarks(imgRGB, a_hand, mp.solutions.hands.HAND_CONNECTIONS)   

        for id, lm in enumerate(a_hand.landmark):
            h, w, c = img.shape

            # 실제 픽셀 추출 
            cx, cy = int(lm.x * w), int( lm.y * h)
            print(id, cx, cy)
            if (id == 0):
                cv2.circle(imgRGB, (cx, cy), 10, (255,0,255), cv2.FILLED)


    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", imgBGR)
    # 'q' 키를 누르면 나감
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break