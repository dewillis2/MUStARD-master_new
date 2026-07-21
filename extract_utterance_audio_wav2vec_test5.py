import json
from pathlib import Path

import torch
import torchaudio
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm


# ===== 路径设置 =====
SAMPLES_PATH = Path(r"C:\Users\17962\Desktop\MUStARD-master\data\processed_samples.json")

# utterance 视频文件夹
UTTERANCE_VIDEO_DIR = Path(r"C:\Users\17962\Desktop\utterances_final")

# 临时 wav 保存文件夹
TEMP_WAV_DIR = Path(r"C:\Users\17962\Desktop\MUStARD-master\new\temp_wav_utterance")

# 输出特征文件，只保存前5个样本
OUTPUT_PATH = Path(r"C:\Users\17962\Desktop\MUStARD-master\new\utterance_audio_wav2vec.pt")

MODEL_NAME = "facebook/wav2vec2-base"


def extract_audio_from_video(video_path: Path, wav_path: Path) -> bool:
    """
    从 mp4 视频中提取音频，保存为 16kHz wav。
    """
    clip = VideoFileClip(str(video_path))

    if clip.audio is None:
        clip.close()
        return False

    clip.audio.write_audiofile(
        str(wav_path),
        fps=16000,
        verbose=False,
        logger=None
    )

    clip.close()
    return True


def main():
    TEMP_WAV_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 读取样本信息
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("总样本数:", len(samples))
    print("本次只测试前5个样本")

    # 加载 wav2vec2
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    features = {}
    missing = []
    failed = []

    with torch.no_grad():
        for sample in tqdm(samples):
            sample_id = sample["id"]
            label = sample["label"]

            video_path = UTTERANCE_VIDEO_DIR / f"{sample_id}.mp4"
            wav_path = TEMP_WAV_DIR / f"{sample_id}.wav"

            if not video_path.exists():
                print("视频不存在:", video_path)
                missing.append(sample_id)
                continue

            try:
                # 1. 从视频中提取音频
                ok = extract_audio_from_video(video_path, wav_path)
                if not ok:
                    print("视频没有音频:", sample_id)
                    failed.append(sample_id)
                    continue

                # 2. 读取 wav 文件
                waveform, sr = torchaudio.load(str(wav_path))

                # 3. 如果是双声道，转成单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # 4. wav2vec2 要求采样率为 16000
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                    sr = 16000

                # 5. 输入 wav2vec2
                inputs = processor(
                    waveform.squeeze(0),
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

                # outputs.last_hidden_state: [1, audio_len, 768]
                # 去掉 batch 维度后: [audio_len, 768]
                audio_feature = outputs.last_hidden_state.squeeze(0).cpu()

                features[sample_id] = {
                    "audio_feature": audio_feature,
                    "label": label,
                    "video_path": str(video_path)
                }

                print(
                    f"\n样本 {sample_id} 处理完成 | "
                    f"audio_feature shape: {audio_feature.shape} | "
                    f"label: {label}"
                )

            except Exception as e:
                print("处理失败:", sample_id)
                print("错误信息:", e)
                failed.append(sample_id)

    # 保存前5个样本的 wav2vec 特征
    torch.save(features, OUTPUT_PATH)

    print("\n========== 处理结束 ==========")
    print("保存完成:", OUTPUT_PATH)
    print("成功数量:", len(features))
    print("缺失视频数量:", len(missing))
    print("失败数量:", len(failed))

    if len(features) > 0:
        first_id = list(features.keys())[0]
        print("\n第一个成功样本:", first_id)
        print("audio feature shape:", features[first_id]["audio_feature"].shape)
        print("label:", features[first_id]["label"])

    if missing:
        print("缺失样本:", missing)

    if failed:
        print("失败样本:", failed)


if __name__ == "__main__":
    main()