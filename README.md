# IAP_FINAL
Final project for IAP

음성 기반 부를 노래 추천 시스템입니다. 사용자가 녹음한 목소리와 설문조사 결과를 학습하여 노래방에서 부를 노래를 추천합니다.  
FFmpeg로 음성을 전처리하고, ONNX 모델(VGGish)로 임베딩 벡터를 추출한 후, MLPClassifier 기반 추천을 수행합니다.


# 디렉토리 구성

PC/
- `getCCBY.ipynb` : 학습할 노래 수집 및 음성파일 전처리 코드  
- `model_on_PC.ipynb` : 사용자 및 노래 음성파일 임베딩 코드  
- `Regression.ipynb` : 모델 학습 및 추출 코드  
- `source/`  
  - `model/` : ONNX 모델 및 추론 코드  
  - `voices/` : 사용자 목소리 (`.wav`)  
  - `music/` : 학습한 노래(`.wav`) — 저작권 문제로 업로드하지 않음  
  - `vectors/` : 임베딩 벡터 (`.npz`)  
nano/
- `regression_new.py` : 추천곡 추론 코드  
- `preprocess.py` : 음성파일 전처리 및 임베딩 코드  
- `source/`  
  - `model/` : ONNX 모델 및 추론 코드  
  - `voices/` : 사용자 목소리 (`.wav`)  
노래방 선호곡 조사 구글폼/ : 설문조사한 선호곡 리스트와 학습 음성파일 
Survey2Vec.ipynb : 설문조사 결과 임베딩 코드



## 사용 방법
1. 젯슨 나노 cmd에서 설치
  wget https://github.com/JUN3-6/IAP_FINAL/archive/refs/heads/main.zip
  unzip main.zip
  mv IAP_FINAL-main/IAP_Final/nano ./nano
  rm -rf IAP_FINAL-main main.zip

3. VGGish 모델 설치
  mkdir -p ./model/VGGish
  wget -O ./model/VGGish/audioset-vggish-3.onnx https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.onnx

3. 이하 디렉토리에 음성 녹음
   `nano/source/test/voice`
   
4. regression.py 실행
   `python3 regression.py`
