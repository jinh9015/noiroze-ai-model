# 베이스 이미지로 시작하십시오
FROM tensorflow/tensorflow:latest

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 종속성 설치
RUN pip install librosa boto3 flask

# 훈련된 모델 추가
COPY all_batch32_dense(224,224).hdf5 /opt/ml/model/all_batch32_dense(224,224).hdf5

# 이전에 작성한 파이썬 스크립트 추가 (new serve.py)
COPY serve.py /app/serve.py

# 람다 함수 호출 액션을 serve 명령어로 지정
ENTRYPOINT ["python", "/app/serve.py"]
