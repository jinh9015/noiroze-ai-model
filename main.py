# 1. 라이브러리 임포트
import os
import librosa
import numpy as np
import tensorflow as tf
import boto3
from tensorflow.keras.models import load_model
from keras.optimizers import Adam

# 2. 모델 불러오기
model = load_model("/opt/ml/model/all_batch32_dense(224,224).hdf5", custom_objects={"Adam": Adam})

class_names = ['1-1어른발걸음', '1-2아이발걸음', '1-3망치질', '2-1가구끄는', '2-2문여닫는','2-3.런닝머신','2-4골프퍼팅','3-1화장실',
               '3-2샤워','3-3드럼세탁기','3-4통돌이세탁기','3-5진공청소기','3-6식기세척기','4-1바이올린','4-2피아노','5-1개','5-2고양이']

check = [0] * len(class_names)  # 가장 많이 나온 것 판단

n_fft = 2048
hop_length = 512
n_mels = 64
fmin = 20
fmax = 8000
duration = 15

def lambda_handler(event, context):
    # 3. 이미지화 -> 판단
    for record in event['Records']:
        # S3 이벤트에서 파일 경로 가져오기
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        wav_file = f'/tmp/{key}'  # Lambda 함수의 임시 디렉토리에 저장할 경로
        
        # S3에서 파일 다운로드
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, wav_file)
        
        # 오디오 파일 불러오기
        y, sr = librosa.load(wav_file)
        
        for j in range(0, len(y) - duration * sr + 1, duration * sr):
            y_interval = y[j:j + duration * sr]
    
            S = librosa.feature.melspectrogram(y=y_interval, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                               n_mels=n_mels, fmin=fmin, fmax=fmax)
    
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB_norm = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB)) * 255
    
            S_dB_norm_resized = tf.image.resize(tf.expand_dims(tf.convert_to_tensor(S_dB_norm), axis=-1),
                                                [224, 224])
            S_dB_norm_resized_4d = tf.expand_dims(S_dB_norm_resized, axis=0)
            S_dB_norm_resized_4d = tf.repeat(S_dB_norm_resized_4d, 3, axis=-1).numpy()
    
            preds = model.predict(S_dB_norm_resized_4d)  # 판단하는 부분
    
            check[np.argmax(preds[0])] += 1
    
        # 4. 결과값
        result_class = class_names[np.argmax(check)]
        print(result_class)
        
        # 결과 데이터를 다시 S3에 업로드하거나 MySQL에 저장하는 로직 추가

        # 업로드된 파일 삭제
        os.remove(wav_file)
        # 생성된 멜 스펙토그램 이미지 삭제
        tf.keras.backend.clear_session()
