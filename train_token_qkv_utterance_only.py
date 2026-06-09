import os

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


TEXT_PATH = Path(r"E:\mustard\MUStARD-master\new\text_token_features.pt")
UTTER_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_audio_wav2vec.pt")
UTTER_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_video_resnet.pt")

SAVE_PATH = Path(r"E:\mustard\MUStARD-master\new\best_token_qkv_utterance_only.pt")

BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
SEED = 42

# 先限制长度，保证能跑
MAX_UTTER_AUDIO_LEN = 250
MAX_UTTER_VIDEO_LEN = 12

# 如果还担心内存，可以先设成 200；如果想全量就改成 None
MAX_SAMPLES = None


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncate_seq(x, max_len):
    if x.shape[0] > max_len:
        return x[:max_len]
    return x


class UtteranceQKVDataset(Dataset):
    def __init__(self):
        print("正在读取 text token features...")
        text = torch.load(TEXT_PATH)

        print("正在读取 utterance audio features...")
        ua = torch.load(UTTER_AUDIO_PATH)

        print("正在读取 utterance video features...")
        uv = torch.load(UTTER_VIDEO_PATH)

        common_ids = set(text.keys()) & set(ua.keys()) & set(uv.keys())
        common_ids = sorted(common_ids)

        if MAX_SAMPLES is not None:
            common_ids = common_ids[:MAX_SAMPLES]

        self.items = []

        for sid in common_ids:
            self.items.append({
                "id": sid,
                "utterance_tokens": text[sid]["utterance_tokens"].float(),
                "utterance_mask": text[sid]["utterance_mask"].long(),
                "utterance_audio": truncate_seq(
                    ua[sid]["audio_feature"].float(),
                    MAX_UTTER_AUDIO_LEN
                ),
                "utterance_video": truncate_seq(
                    uv[sid]["video_feature"].float(),
                    MAX_UTTER_VIDEO_LEN
                ),
                "label": torch.tensor(text[sid]["label"], dtype=torch.long)
            })

        del text
        del ua
        del uv
        gc.collect()

        print("可用样本数:", len(self.items))

        x = self.items[0]
        print("示例ID:", x["id"])
        print("utterance_tokens:", x["utterance_tokens"].shape)
        print("utterance_audio:", x["utterance_audio"].shape)
        print("utterance_video:", x["utterance_video"].shape)
        print("label:", x["label"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def pad_seq(seqs):
    lengths = torch.tensor([x.shape[0] for x in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    max_len = padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask


def collate_fn(batch):
    utter_tokens = torch.stack([x["utterance_tokens"] for x in batch], dim=0)
    utter_mask = torch.stack([x["utterance_mask"] for x in batch], dim=0).bool()

    utter_audio, utter_audio_mask = pad_seq([x["utterance_audio"] for x in batch])
    utter_video, utter_video_mask = pad_seq([x["utterance_video"] for x in batch])

    labels = torch.stack([x["label"] for x in batch], dim=0)

    return {
        "utterance_tokens": utter_tokens,
        "utterance_mask": utter_mask,
        "utterance_audio": utter_audio,
        "utterance_audio_mask": utter_audio_mask,
        "utterance_video": utter_video,
        "utterance_video_mask": utter_video_mask,
        "label": labels
    }


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class UtteranceQKVCrossAttentionModel(nn.Module):
    """
    真正的 Q-K-V cross-modal attention。

    Text-Audio:
        Q = BERT token-level utterance text
        K,V = wav2vec2 utterance audio sequence

    Text-Video:
        Q = BERT token-level utterance text
        K,V = ResNet utterance video frame sequence
    """

    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()

        self.text_proj = nn.Linear(768, hidden_dim)
        self.audio_proj = nn.Linear(768, hidden_dim)
        self.video_proj = nn.Linear(2048, hidden_dim)

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

        self.text_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.video_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def cross_attend_audio(self, text_tokens, text_mask, audio_seq, audio_mask):
        q = self.text_norm(self.text_proj(text_tokens))   # [B, T, H]
        k = self.audio_norm(self.audio_proj(audio_seq))   # [B, A, H]
        v = k

        out, attn_weights = self.text_audio_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=~audio_mask
        )

        out = masked_mean(out, text_mask)
        return out

    def cross_attend_video(self, text_tokens, text_mask, video_seq, video_mask):
        q = self.text_norm(self.text_proj(text_tokens))   # [B, T, H]
        k = self.video_norm(self.video_proj(video_seq))   # [B, V, H]
        v = k

        out, attn_weights = self.text_video_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=~video_mask
        )

        out = masked_mean(out, text_mask)
        return out

    def forward(self, batch):
        text_tokens = batch["utterance_tokens"]
        text_mask = batch["utterance_mask"]

        audio_seq = batch["utterance_audio"]
        audio_mask = batch["utterance_audio_mask"]

        video_seq = batch["utterance_video"]
        video_mask = batch["utterance_video_mask"]

        text_h = masked_mean(
            self.text_norm(self.text_proj(text_tokens)),
            text_mask
        )

        audio_h = self.cross_attend_audio(
            text_tokens,
            text_mask,
            audio_seq,
            audio_mask
        )

        video_h = self.cross_attend_video(
            text_tokens,
            text_mask,
            video_seq,
            video_mask
        )

        fused = torch.cat([text_h, audio_h, video_h], dim=1)

        logits = self.classifier(fused)
        return logits


def move_batch_to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


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


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = UtteranceQKVDataset()

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
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = UtteranceQKVCrossAttentionModel(
        hidden_dim=128,
        num_heads=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0
    best_result = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            labels = batch["label"]

            logits = model(batch)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

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

    print("\nBest result:")
    print(best_result)


if __name__ == "__main__":
    main()