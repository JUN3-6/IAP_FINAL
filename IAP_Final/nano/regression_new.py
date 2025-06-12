import numpy as np
import onnxruntime as ort
from preprocess import extract_highlight, embedding

# 음성 전처리
extract_highlight()
embedding()

# 벡터 불러오기
voices_npz = np.load("./source/test/voice_vectors.npz")
songs_npz  = np.load("./source/vectors/music_vectors.npz")

M = len(voices_npz.files)
N = len(songs_npz.files)
voice_keys = voices_npz.files[:M]
song_keys  = songs_npz.files[:N]

print(f"{M}명의 목소리에 대해 {N}곡 중 추천하는 노래 탐색중")

# 입력 벡터 구성
voices = np.stack([voices_npz[k] for k in voice_keys])   # (M, 128)
songs  = np.stack([songs_npz[k] for k in song_keys])     # (N, 128)

voice_repeated = np.repeat(voices, N, axis=0)            # (M×N, 128)
song_tiled     = np.tile(songs, (M, 1))                  # (M×N, 128)
X              = np.hstack([voice_repeated, song_tiled]) # (M×N, 256)

# ONNX 모델 로드
onnx_path = "./source/model/MLP/model.onnx"
sess = ort.InferenceSession(onnx_path)

# 입력 이름 확인
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

# 추론
outputs = sess.run(output_names, {input_name: X.astype(np.float32)})
y_pred = outputs[0]
y_prob = outputs[1][:, 1]


# 상위 K개 추천 출력
K = 5
recommend = list(song_keys)

for i, voice_id in enumerate(voice_keys):
    start = i * N
    end   = start + N
    probs = y_prob[start:end]
    preds = y_pred[start:end]

    top_idx = np.argsort(probs)[::-1][:K]

    name = voice_id.replace("cliped_","")

    print(f"\n== 음성 '{name}' 에 대한 추천 Top-{K} ==")
    for rank, idx in enumerate(top_idx, 1):
        song_name = recommend[idx].replace("cliped_","")
        print(f"{rank}. {song_name} | 확률: {probs[idx]*100:.2f}% | 예측: {'좋음' if preds[idx]==1 else '나쁨'}")
