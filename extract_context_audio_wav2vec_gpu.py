import os

# 必须放在 transformers 导入之前，防止模型缓存到 C 盘
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache"
os.environ["TORCH_HOME"] = r"E:\torch_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
from pathlib import Path

import torch
import torchaudio
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm


# ===== 路径设置 =====
SAMPLES_PATH = Path(r"E:\mustard\MUStARD-master\data\processed_samples.json")
CONTEXT_VIDEO_DIR = Path(r"E:\sarcasm_vidio\context_final")

TEMP_WAV_DIR = Path(r"E:\mustard\MUStARD-master\new\temp_wav_context")
OUTPUT_PATH = Path(r"E:\mustard\MUStARD-master\new\context_audio_wav2vec.pt")

MODEL_NAME = "facebook/wav2vec2-base"


def extract_audio_from_video(video_path: Path, wav_path: Path) -> bool:
    """
    用 moviepy 从 context 视频中提取 wav 音频。
    如果 wav 已经存在，就不重复提取。
    """
    if wav_path.exists():
        return True

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

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("样本文件:", SAMPLES_PATH)
    print("context视频文件夹:", CONTEXT_VIDEO_DIR)
    print("临时wav文件夹:", TEMP_WAV_DIR)
    print("输出文件:", OUTPUT_PATH)

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("总样本数:", len(samples))

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # 断点续跑：如果已有输出文件，就继续在原文件基础上跑
    if OUTPUT_PATH.exists():
        features = torch.load(OUTPUT_PATH)
        print("检测到已有特征文件，已完成数量:", len(features))
    else:
        features = {}

    missing = []
    failed = []

    with torch.no_grad():
        for sample in tqdm(samples):
            sample_id = sample["id"]
            label = sample["label"]

            # 已经处理过就跳过
            if sample_id in features:
                continue

            # context 视频文件名是 1_60_c.mp4
            video_path = CONTEXT_VIDEO_DIR / f"{sample_id}_c.mp4"
            wav_path = TEMP_WAV_DIR / f"{sample_id}_c.wav"

            if not video_path.exists():
                print("context 视频不存在:", video_path)
                missing.append(sample_id)
                continue

            try:
                # 1. 从 mp4 提取 wav
                ok = extract_audio_from_video(video_path, wav_path)
                if not ok:
                    print("音频提取失败:", sample_id)
                    failed.append(sample_id)
                    continue

                # 2. 读取 wav
                waveform, sr = torchaudio.load(str(wav_path))

                # 3. 转单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # 4. wav2vec2 要求 16000Hz
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

                # [1, audio_len, 768] -> [audio_len, 768]
                context_audio_feature = outputs.last_hidden_state.squeeze(0).cpu()

                features[sample_id] = {
                    "context_audio_feature": context_audio_feature,
                    "label": label,
                    "video_path": str(video_path)
                }

                # 每处理一个样本就保存一次，防止中途断掉白跑
                torch.save(features, OUTPUT_PATH)

            except Exception as e:
                print("处理失败:", sample_id)
                print("错误信息:", e)
                failed.append(sample_id)

    torch.save(features, OUTPUT_PATH)

    print("\n========== 处理结束 ==========")
    print("保存完成:", OUTPUT_PATH)
    print("成功数量:", len(features))
    print("缺失视频数量:", len(missing))
    print("失败数量:", len(failed))

    if len(features) > 0:
        first_id = list(features.keys())[0]
        print("第一个样本:", first_id)
        print("context audio feature shape:", features[first_id]["context_audio_feature"].shape)
        print("label:", features[first_id]["label"])

    if missing:
        print("缺失前10个:", missing[:10])

    if failed:
        print("失败前10个:", failed[:10])


if __name__ == "__main__":
    main()