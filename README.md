# 팀프로젝트(3인) - 감정분석을 통한 졸음운전 감지 모델

## 개요
🔍운전자의 표정을 객체 감지 모델로 분석하여 졸음운전의 유무를 감지해서 방지하는 시스템

🔍다양한 생체 신호와 행동패턴을 종합적으로 고려하여 졸음 운전 시스템의 통합적 분석에 기여를 목표

## 기술 스택
|기술|사용|
|-----|-------|
|cv2|openCV 라이브러리|
|dlib|얼굴 랜드마크 탐지|
|Numpy|다차원 배열 및 행렬에 사용|
|Pillow|바운딩박스 사용, 텍스트 및 도형 추가|

## 문제점 해결
### 📝문제. 하나의 감정 수치를 조정하였을 경우, 기존의 값과 겹치는 곳이 있어 인식 오류 발생

#### 🔍해결
얼굴의 특정 위치(눈썹과 눈 사이, 입꼬리 각도 등) 나타내는 코드를 조정하고,

다른 값들과 수치 비교하여 해당 감정만 나타날 수 있도록 수정

-----------------------------
#### 🔍코드 첨부

  🔹수정 전 (눈썹의 위치만 조정하니 정확도 낮음)
     
<img src = "https://github.com/user-attachments/assets/8d9ca092-f224-4671-94d4-40cf20d1580b" width="300" height="200">

  🔹수정 후 (양쪽 눈 높이와 눈썹과 눈 사이의 거리측정코드 추가 -> 감정'기쁨'을 감지하는 속도 빨라짐)

<img src = "https://github.com/user-attachments/assets/a191e800-5b6c-4521-b525-c5c361835a0b" width="300" height="200">


   
