import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time  # 시간 측정을 위한 모듈

save_dir = './pose_img/person/'
dataset_file = f'{save_dir}dataset.csv'
weight_file = f'{save_dir}model_weights.xgb'

# 데이터셋 로드
df = pd.read_csv(dataset_file)

# 특징(X)과 타겟(y) 변수 정의
X = df.drop(['label', 'image_name'], axis=1)  # 'label' 컬럼이 타겟 변수라고 가정
y = df['label'].map({'sitting': 0, 'standing': 1})  # 레이블을 0과 1로 변환

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost 분류기 생성
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# 학습 시작 시간 기록
start_time = time.time()
print("Training started...")

# 모델 학습
model.fit(X_train, y_train)

# 학습 종료 시간 기록
end_time = time.time()
print("Training finished...")

# 총 소요시간 계산
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")

# 테스트 세트에 대해 예측 수행
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 학습된 모델 저장
model.save_model(weight_file)
