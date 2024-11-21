from ultralytics import YOLO
import cv2
import xgboost as xgb
import pandas as pd

save_dir = './pose_img/person/'
weight_file = f'{save_dir}model_weights.xgb'

# YOLOv8 모델 로드
model_yolo = YOLO('yolov8n-pose.pt')

# XGBoost 모델 로드 (Booster 대신 XGBClassifier 사용)
model = xgb.XGBClassifier()
model.load_model(weight_file)

# 카메라에서 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# 총 프레임 수 출력 (일부 카메라에서는 0일 수 있음)
print('Total Frame', cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_tot = 0  # 처리된 프레임 수

# 비디오 프레임 반복 처리
while cap.isOpened():
    # 비디오 프레임 읽기
    success, frame = cap.read()

    if success:
        # YOLOv8로 프레임에서 객체 감지 수행
        results = model_yolo(frame, verbose=False)

        # 결과를 시각화하여 프레임에 그리기
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            bound_box = r.boxes.xyxy  # 경계 상자 정보
            conf = r.boxes.conf.tolist()  # 신뢰도 값
            keypoints = r.keypoints.xyn.tolist()  # 키포인트 좌표

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:  # 신뢰도 0.75 이상일 때만 처리
                    x1, y1, x2, y2 = box.tolist()
                    data = {}

                    # 키포인트 좌표 저장
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    # 데이터프레임 생성 (DMatrix 대신 사용)
                    df = pd.DataFrame([data])
                    
                    # XGBoost 모델을 이용해 예측 수행 (DMatrix 없이)
                    cut = model.predict(df)
                    binary_predictions = (cut > 0.5).astype(int)

                    # 예측 결과가 'sitting'인 경우 파란색 사각형 그리기
                    if binary_predictions == 0:
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, 'Sitting', (int(x1), int(y1)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # 주석이 달린 프레임을 화면에 표시 (창 이름 필수)
        cv2.imshow("Sitting Action Recognition", annotated_frame)

        frame_tot += 1  # 처리된 프레임 수 증가
        # print('Processed Frame : ', frame_tot)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 비디오 끝에 도달하면 종료
        break

# 비디오 캡처 객체 및 창 닫기
cap.release()
cv2.destroyAllWindows()
