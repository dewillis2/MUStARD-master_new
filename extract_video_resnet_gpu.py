import os

# 防止模型缓存到 C 盘
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache"
os.environ["TORCH_HOME"] = r"E:\torch_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from moviepy.editor import VideoFileClip
from PIL import Image
from tqdm import tqdm


# ===== 路径设置 =====
SAMPLES_PATH = Path(r"E:\mustard\MUStARD-master\data\processed_samples.json")

UTTERANCE_VIDEO_DIR = Path(r"E:\sarcasm_vidio\utterances_final")
CONTEXT_VIDEO_DIR = Path(r"E:\sarcasm_vidio\context_final")

UTTERANCE_OUTPUT_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_video_resnet.pt")
CONTEXT_OUTPUT_PATH = Path(r"E:\mustard\MUStARD-master\new\context_video_resnet.pt")

# 每秒抽几帧。1 表示每秒 1 帧
SAMPLE_FPS = 1

# 每个视频最多抽多少帧，防止个别视频太长
MAX_FRAMES = 32


def build_resnet_feature_extractor(device):
    weights = ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)

    # 去掉最后的分类层，只保留到 global average pooling
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    transform = weights.transforms()

    return feature_extractor, transform


def extract_video_feature(video_path: Path, model, transform, device):
    """
    输入一个 mp4 视频，输出 [frame_len, 2048] 的 ResNet 特征。
    """
    frames = []

    clip = VideoFileClip(str(video_path))

    try:
        for frame in clip.iter_frames(fps=SAMPLE_FPS, dtype="uint8"):
            img = Image.fromarray(frame).convert("RGB")
            img_tensor = transform(img)
            frames.append(img_tensor)

            if len(frames) >= MAX_FRAMES:
                break
    finally:
        clip.close()

    if len(frames) == 0:
        return None

    batch = torch.stack(frames, dim=0).to(device)

    with torch.no_grad():
        feats = model(batch)

    # [frame_len, 2048, 1, 1] -> [frame_len, 2048]
    feats = feats.squeeze(-1).squeeze(-1).cpu()

    return feats


def load_existing(path: Path):
    if path.exists():
        data = torch.load(path)
        print(f"检测到已有文件: {path}")
        print("已完成数量:", len(data))
        return data
    return {}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("样本文件:", SAMPLES_PATH)
    print("utterance视频文件夹:", UTTERANCE_VIDEO_DIR)
    print("context视频文件夹:", CONTEXT_VIDEO_DIR)

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("总样本数:", len(samples))

    model, transform = build_resnet_feature_extractor(device)

    utterance_features = load_existing(UTTERANCE_OUTPUT_PATH)
    context_features = load_existing(CONTEXT_OUTPUT_PATH)

    utter_missing = []
    context_missing = []
    utter_failed = []
    context_failed = []

    for sample in tqdm(samples):
        sample_id = sample["id"]
        label = sample["label"]

        # ===== utterance video =====
        if sample_id not in utterance_features:
            utter_video_path = UTTERANCE_VIDEO_DIR / f"{sample_id}.mp4"

            if not utter_video_path.exists():
                utter_missing.append(sample_id)
            else:
                try:
                    feat = extract_video_feature(
                        utter_video_path,
                        model,
                        transform,
                        device
                    )

                    if feat is None:
                        utter_failed.append(sample_id)
                    else:
                        utterance_features[sample_id] = {
                            "video_feature": feat,
                            "label": label,
                            "video_path": str(utter_video_path)
                        }

                        torch.save(utterance_features, UTTERANCE_OUTPUT_PATH)

                except Exception as e:
                    print("utterance video 处理失败:", sample_id, e)
                    utter_failed.append(sample_id)

        # ===== context video =====
        if sample_id not in context_features:
            context_video_path = CONTEXT_VIDEO_DIR / f"{sample_id}_c.mp4"

            if not context_video_path.exists():
                context_missing.append(sample_id)
            else:
                try:
                    feat = extract_video_feature(
                        context_video_path,
                        model,
                        transform,
                        device
                    )

                    if feat is None:
                        context_failed.append(sample_id)
                    else:
                        context_features[sample_id] = {
                            "context_video_feature": feat,
                            "label": label,
                            "video_path": str(context_video_path)
                        }

                        torch.save(context_features, CONTEXT_OUTPUT_PATH)

                except Exception as e:
                    print("context video 处理失败:", sample_id, e)
                    context_failed.append(sample_id)

    torch.save(utterance_features, UTTERANCE_OUTPUT_PATH)
    torch.save(context_features, CONTEXT_OUTPUT_PATH)

    print("\n========== 视频特征提取结束 ==========")
    print("utterance 保存:", UTTERANCE_OUTPUT_PATH)
    print("utterance 成功数量:", len(utterance_features))
    print("utterance 缺失数量:", len(utter_missing))
    print("utterance 失败数量:", len(utter_failed))

    print("context 保存:", CONTEXT_OUTPUT_PATH)
    print("context 成功数量:", len(context_features))
    print("context 缺失数量:", len(context_missing))
    print("context 失败数量:", len(context_failed))

    if len(utterance_features) > 0:
        first_id = list(utterance_features.keys())[0]
        print("\nutterance 示例:", first_id)
        print("utterance video feature shape:", utterance_features[first_id]["video_feature"].shape)

    if len(context_features) > 0:
        first_id = list(context_features.keys())[0]
        print("\ncontext 示例:", first_id)
        print("context video feature shape:", context_features[first_id]["context_video_feature"].shape)


if __name__ == "__main__":
    main()