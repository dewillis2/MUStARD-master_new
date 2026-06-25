import os

# ===== 防止缓存写入 C 盘 =====
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache"
os.environ["TORCH_HOME"] = r"E:\torch_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path
import random
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


# =========================================================
# 路径设置
# =========================================================
TEXT_PATH = Path(r"E:\mustard\MUStARD-master\new\text_token_features.pt")

UTTER_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_audio_wav2vec.pt")
CONTEXT_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_audio_wav2vec.pt")

UTTER_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_video_resnet.pt")
CONTEXT_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_video_resnet.pt")

SAVE_PATH = Path(r"E:\mustard\MUStARD-master\new\best_qkv_full_memory_safe.pt")


# =========================================================
# 训练参数
# =========================================================
BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
SEED = 42

# 为了节省显存和内存，先限制序列长度
MAX_UTTER_TEXT_LEN = 64
MAX_CONTEXT_TEXT_LEN = 128

MAX_UTTER_AUDIO_LEN = 200
MAX_CONTEXT_AUDIO_LEN = 250

MAX_UTTER_VIDEO_LEN = 8
MAX_CONTEXT_VIDEO_LEN = 12

# 如果本地还是内存不够，先改成 100 / 200 测试
# 全量就设为 None
MAX_SAMPLES = None

# 模型维度。显存不够就改成 64
HIDDEN_DIM = 128
NUM_HEADS = 4

USE_AMP = True


# =========================================================
# 工具函数
# =========================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    """
    尽量使用 mmap=True 读取，减少 CPU 内存压力。
    如果当前 torch 版本不支持，就自动退回普通 torch.load。
    """
    print(f"读取文件: {path}")
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def truncate_seq(x, max_len):
    """
    x: [seq_len, dim]
    """
    if x.shape[0] > max_len:
        return x[:max_len]
    return x


def pad_seq(seqs):
    """
    输入 list of [seq_len, dim]
    输出:
        padded: [batch, max_len, dim]
        mask:   [batch, max_len]
    mask 中 True 表示真实位置，False 表示 padding。
    """
    lengths = torch.tensor([x.shape[0] for x in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)

    max_len = padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return padded, mask


def masked_mean(x, mask):
    """
    x: [batch, seq_len, hidden_dim]
    mask: [batch, seq_len]
    """
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


# =========================================================
# Dataset
# =========================================================
class FullQKVDataset(Dataset):
    def __init__(self):
        print("正在读取特征文件...")

        self.text = safe_torch_load(TEXT_PATH)
        self.ua = safe_torch_load(UTTER_AUDIO_PATH)
        self.ca = safe_torch_load(CONTEXT_AUDIO_PATH)
        self.uv = safe_torch_load(UTTER_VIDEO_PATH)
        self.cv = safe_torch_load(CONTEXT_VIDEO_PATH)

        common_ids = (
            set(self.text.keys())
            & set(self.ua.keys())
            & set(self.ca.keys())
            & set(self.uv.keys())
            & set(self.cv.keys())
        )

        self.ids = sorted(common_ids)

        if MAX_SAMPLES is not None:
            self.ids = self.ids[:MAX_SAMPLES]

        print("可用样本数:", len(self.ids))

        if len(self.ids) > 0:
            sid = self.ids[0]
            print("示例ID:", sid)
            print("utterance_tokens:", self.text[sid]["utterance_tokens"].shape)
            print("context_tokens:", self.text[sid]["context_tokens"].shape)
            print("utterance_audio:", self.ua[sid]["audio_feature"].shape)
            print("context_audio:", self.ca[sid]["context_audio_feature"].shape)
            print("utterance_video:", self.uv[sid]["video_feature"].shape)
            print("context_video:", self.cv[sid]["context_video_feature"].shape)
            print("label:", self.text[sid]["label"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        # 文本 token
        utter_tokens = self.text[sid]["utterance_tokens"][:MAX_UTTER_TEXT_LEN].float()
        utter_mask = self.text[sid]["utterance_mask"][:MAX_UTTER_TEXT_LEN].long()

        context_tokens = self.text[sid]["context_tokens"][:MAX_CONTEXT_TEXT_LEN].float()
        context_mask = self.text[sid]["context_mask"][:MAX_CONTEXT_TEXT_LEN].long()

        # audio
        utter_audio = truncate_seq(
            self.ua[sid]["audio_feature"],
            MAX_UTTER_AUDIO_LEN
        ).float()

        context_audio = truncate_seq(
            self.ca[sid]["context_audio_feature"],
            MAX_CONTEXT_AUDIO_LEN
        ).float()

        # video
        utter_video = truncate_seq(
            self.uv[sid]["video_feature"],
            MAX_UTTER_VIDEO_LEN
        ).float()

        context_video = truncate_seq(
            self.cv[sid]["context_video_feature"],
            MAX_CONTEXT_VIDEO_LEN
        ).float()

        label = torch.tensor(self.text[sid]["label"], dtype=torch.long)

        return {
            "id": sid,

            "utterance_tokens": utter_tokens,
            "utterance_mask": utter_mask,

            "context_tokens": context_tokens,
            "context_mask": context_mask,

            "utterance_audio": utter_audio,
            "context_audio": context_audio,

            "utterance_video": utter_video,
            "context_video": context_video,

            "label": label
        }


def collate_fn(batch):
    # text 已经固定长度，直接 stack
    utter_tokens = torch.stack([x["utterance_tokens"] for x in batch], dim=0)
    utter_mask = torch.stack([x["utterance_mask"] for x in batch], dim=0).bool()

    context_tokens = torch.stack([x["context_tokens"] for x in batch], dim=0)
    context_mask = torch.stack([x["context_mask"] for x in batch], dim=0).bool()

    # audio / video 变长，需要 pad
    utter_audio, utter_audio_mask = pad_seq([x["utterance_audio"] for x in batch])
    context_audio, context_audio_mask = pad_seq([x["context_audio"] for x in batch])

    utter_video, utter_video_mask = pad_seq([x["utterance_video"] for x in batch])
    context_video, context_video_mask = pad_seq([x["context_video"] for x in batch])

    labels = torch.stack([x["label"] for x in batch], dim=0)

    return {
        "utterance_tokens": utter_tokens,
        "utterance_mask": utter_mask,

        "context_tokens": context_tokens,
        "context_mask": context_mask,

        "utterance_audio": utter_audio,
        "utterance_audio_mask": utter_audio_mask,

        "context_audio": context_audio,
        "context_audio_mask": context_audio_mask,

        "utterance_video": utter_video,
        "utterance_video_mask": utter_video_mask,

        "context_video": context_video,
        "context_video_mask": context_video_mask,

        "label": labels
    }


# =========================================================
# Model
# =========================================================
class FullQKVCrossAttentionModel(nn.Module):
    """
    完整版 Q-K-V cross-modal attention。

    使用：
    1. utterance text -> utterance audio
    2. utterance text -> utterance video
    3. context text -> context audio
    4. context text -> context video

    Q = text token
    K,V = audio sequence / video frame sequence
    """

    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 不同模态投影到同一维度
        self.text_proj = nn.Linear(768, hidden_dim)
        self.audio_proj = nn.Linear(768, hidden_dim)
        self.video_proj = nn.Linear(2048, hidden_dim)

        self.text_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.video_norm = nn.LayerNorm(hidden_dim)

        # Q-K-V attention
        self.text_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.text_video_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 融合向量：
        # utter_text_h
        # context_text_h
        # utter_audio_h
        # context_audio_h
        # utter_video_h
        # context_video_h
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def project_text(self, text_tokens):
        return self.text_norm(self.text_proj(text_tokens))

    def project_audio(self, audio_seq):
        return self.audio_norm(self.audio_proj(audio_seq))

    def project_video(self, video_seq):
        return self.video_norm(self.video_proj(video_seq))

    def cross_attend_audio(self, text_tokens, text_mask, audio_seq, audio_mask):
        """
        Q = text tokens
        K,V = audio sequence

        输出:
        [batch, hidden_dim]
        """
        q = self.project_text(text_tokens)
        k = self.project_audio(audio_seq)
        v = k

        out, _ = self.text_audio_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=~audio_mask,
            need_weights=False
        )

        out = masked_mean(out, text_mask)
        return out

    def cross_attend_video(self, text_tokens, text_mask, video_seq, video_mask):
        """
        Q = text tokens
        K,V = video frames

        输出:
        [batch, hidden_dim]
        """
        q = self.project_text(text_tokens)
        k = self.project_video(video_seq)
        v = k

        out, _ = self.text_video_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=~video_mask,
            need_weights=False
        )

        out = masked_mean(out, text_mask)
        return out

    def forward(self, batch):
        utter_tokens = batch["utterance_tokens"]
        utter_mask = batch["utterance_mask"]

        context_tokens = batch["context_tokens"]
        context_mask = batch["context_mask"]

        utter_audio = batch["utterance_audio"]
        utter_audio_mask = batch["utterance_audio_mask"]

        context_audio = batch["context_audio"]
        context_audio_mask = batch["context_audio_mask"]

        utter_video = batch["utterance_video"]
        utter_video_mask = batch["utterance_video_mask"]

        context_video = batch["context_video"]
        context_video_mask = batch["context_video_mask"]

        # text 自身表示
        utter_text_h = masked_mean(
            self.project_text(utter_tokens),
            utter_mask
        )

        context_text_h = masked_mean(
            self.project_text(context_tokens),
            context_mask
        )

        # Q-K-V cross attention
        utter_audio_h = self.cross_attend_audio(
            utter_tokens,
            utter_mask,
            utter_audio,
            utter_audio_mask
        )

        context_audio_h = self.cross_attend_audio(
            context_tokens,
            context_mask,
            context_audio,
            context_audio_mask
        )

        utter_video_h = self.cross_attend_video(
            utter_tokens,
            utter_mask,
            utter_video,
            utter_video_mask
        )

        context_video_h = self.cross_attend_video(
            context_tokens,
            context_mask,
            context_video,
            context_video_mask
        )

        fused = torch.cat(
            [
                utter_text_h,
                context_text_h,
                utter_audio_h,
                context_audio_h,
                utter_video_h,
                context_video_h
            ],
            dim=1
        )

        logits = self.classifier(fused)
        return logits


# =========================================================
# Evaluate
# =========================================================
def evaluate(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch["label"]

            logits = model(batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    correct = sum(p == y for p, y in zip(all_preds, all_labels))
    acc = correct / total if total > 0 else 0

    tp = sum((p == 1 and y == 1) for p, y in zip(all_preds, all_labels))
    fp = sum((p == 1 and y == 0) for p, y in zip(all_preds, all_labels))
    fn = sum((p == 0 and y == 1) for p, y in zip(all_preds, all_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return acc, precision, recall, f1


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = FullQKVDataset()

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = FullQKVCrossAttentionModel(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    best_f1 = 0.0
    best_result = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            labels = batch["label"]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                logits = model(batch)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch:02d} | Step {step} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        acc, precision, recall, f1 = evaluate(
            model,
            test_loader,
            device
        )

        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                "epoch": epoch,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            torch.save(model.state_dict(), SAVE_PATH)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"P: {precision:.4f} | "
            f"R: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

    print("\nBest result:")
    print(best_result)
    print("Best model saved to:", SAVE_PATH)


if __name__ == "__main__":
    main()