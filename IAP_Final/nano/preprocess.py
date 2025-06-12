import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

import torch
import torchaudio
import torchaudio.transforms as T

import shutil
import onnxruntime

import re


def extract_highlight(input_dir="source/test/voice", output_dir="source/test/voice_clips", clip_duration=15.0):
    print("Extracting 15-second highlight segments...")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    print(f"Found {len(files)} files to process.")

    for f in tqdm(files):
        input_path = os.path.join(input_dir, f)
        y, sr = librosa.load(input_path, sr=None)

        total_duration = len(y) / sr
        if total_duration < clip_duration:
            print(f"{f} is shorter than {clip_duration} seconds. Skipping.")
            continue

        clip_len_samples = int(clip_duration * sr)
        hop_length = sr

        max_energy = 0
        best_start = 0

        for start in range(0, len(y) - clip_len_samples, hop_length):
            segment = y[start:start + clip_len_samples]
            energy = np.sum(segment ** 2)
            if energy > max_energy:
                max_energy = energy
                best_start = start

        highlight = y[best_start:best_start + clip_len_samples]

        name_only = os.path.splitext(f)[0]
        trimmed_name = name_only[len("voice_only_"):] if name_only.startswith("voice_only_") else name_only
        output_filename = f"cliped_{trimmed_name}.wav"
        output_path = os.path.join(output_dir, output_filename)

        sf.write(output_path, highlight, sr)

    print("Highlight extraction complete.")

def preprocess_for_vggish(wav_path, target_sample_rate=16000):
    # 1. 오디오 로드 및 리샘플링
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sample_rate:
        resampler = T.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    # 2. 채널 평균 (Mono로 변환)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3. Mel Spectrogram (torchaudio의 설정은 VGGish에 맞게)
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=64,
        f_min=125,
        f_max=7500,
        power=2.0
    )
    mel = mel_transform(waveform)  # (1, 64, T)

    # 4. dB 변환
    db_transform = T.AmplitudeToDB()
    mel_db = db_transform(mel)  # (1, 64, T)

    # 5. Frame 단위 자르기 (96x64)
    mel_db = mel_db.squeeze(0).transpose(0, 1)  # (T, 64)
    if mel_db.shape[0] < 96:
        # 패딩
        pad_len = 96 - mel_db.shape[0]
        mel_db = torch.nn.functional.pad(mel_db, (0, 0, 0, pad_len))
    else:
        mel_db = mel_db[:96, :]  # 앞에서부터 96 프레임 자름
    mel_db = mel_db.transpose(0, 1)
    mel_db = mel_db.unsqueeze(0)  # (1, 1, 96, 64)
    return mel_db

def run_vggish_onnx(sess, mel_db):
    """
    mel_db: torch.Tensor of shape (1, 1, 96, 64)
    onnx_path: path to vggish.onnx
    """

    mel_spec_np = mel_db.numpy().astype(np.float32)

    # 입력 이름 확인
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # 추론
    outputs = sess.run([output_name], {input_name: mel_spec_np})
    embedding = outputs[0]  # shape: (1, 128)

    return embedding.squeeze()

def embedding(input_dir = "source/test/voice_clips", output_npz = "source/test/voice_vectors.npz", key_path=""):
# 설정
    onnx_path = "source/model/VGGish/audioset-vggish-3.onnx"

    # 기존 벡터 불러오기 (있다면)
    audio_vectors = {}
    if os.path.exists(output_npz):
        existing = np.load(output_npz, allow_pickle=True)
        audio_vectors.update({k: existing[k] for k in existing.files})
        print(f"✅ 기존 벡터 {len(audio_vectors)}개 로드됨")

    # ONNX 세션 생성
    sess = onnxruntime.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider", "AzureExecutionProvider"]
    )

    # 벡터 추출 루프
    def order_key(reference_list_path=None):
        reference_order = {}
        if reference_list_path and os.path.exists(reference_list_path):
            with open(reference_list_path, "r", encoding="utf-8") as f:
                titles = [line.strip() for line in f]
                for idx, title in enumerate(titles):
                    filename = f"cliped_{title}.wav"
                    reference_order[filename] = idx

        def sort_key(filename):
            if filename in reference_order:
                return reference_order[filename]
            match = re.search(r"\((\d+)\)", filename)
            return int(match.group(1)) if match else float('inf')

        return sort_key

    for fname in sorted(os.listdir(input_dir), key=order_key(key_path)):
        if not fname.lower().endswith(".wav"):
            continue

        input_path = os.path.join(input_dir, fname)

        try:
            key = os.path.splitext(fname)[0]

            mel_db = preprocess_for_vggish(input_path, target_sample_rate=16000)
            audio_vectors[key] = run_vggish_onnx(sess, mel_db)

            moved_dir = "./"+input_dir+"/loaded_musics"
            os.makedirs(moved_dir, exist_ok=True)
            shutil.move(input_path, os.path.join(moved_dir, fname))

            print(f"✅ 처리 완료: {fname}")

        except Exception as e:
            print(f"❌ 실패: {fname} - {e}")

    # .npz 저장
    np.savez(output_npz, **audio_vectors)
    print(f"\n💾 총 {len(audio_vectors)}개의 벡터 저장 완료 → {output_npz}")
