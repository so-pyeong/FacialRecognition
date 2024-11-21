import cv2
from ultralytics import YOLO

# 1. YOLOv8 Pose 모델 불러오기 (미리 학습된 모델 사용)
model = YOLO('yolov8n-pose.pt')  # 'n'은 모델 크기 옵션 (n, s, m, l, x)

# 2. 실시간 카메라 비디오 스트림 열기
cap = cv2.VideoCapture(0)  # 0번 카메라 열기

# 3. 비디오 스트림에서 프레임을 받아와서 포즈 추정
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO Pose 모델로 포즈 추정 수행
    results = model(frame)

    # 추정된 포즈를 그리기
    annotated_frame = results[0].plot()  # 추정된 포즈를 프레임 위에 그림

    # 4. 결과를 화면에 표시
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 카메라와 창 닫기
cap.release()
cv2.destroyAllWindows()