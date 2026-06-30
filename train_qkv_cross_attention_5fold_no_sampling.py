import os
import gc
import csv
import random
import warnings
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# =========================
# 基本设置
# =========================

BASE_DIR = r"E:\mustard\MUStARD-master\new"

TEXT_PATH = os.path.join(BASE_DIR, "text_token_features.pt")
UTTER_AUDIO_PATH = os.path.join(BASE_DIR, "utterance_audio_wav2vec.pt")
CONTEXT_AUDIO_PATH = os.path.join(BASE_DIR, "context_audio_wav2vec.pt")
UTTER_VIDEO_PATH = os.path.join(BASE_DIR, "utterance_video_resnet.pt")
CONTEXT_VIDEO_PATH = os.path.join(BASE_DIR, "context_video_resnet.pt")

SAVE_DIR = BASE_DIR

SEED = 42
N_SPLITS = 5
MAX_EPOCHS = 10
BATCH_SIZE = 1

TEXT_DIM = 768
AUDIO_DIM = 768
VIDEO_DIM = 2048

# 你现在全系列版用的是 64，先保持一致
HIDDEN_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.4

LR = 1e-4
WEIGHT_DECAY = 1e-4

USE_AMP = True
EARLY_STOPPING_PATIENCE = 3


warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def safe_torch_load(path):
    print(f"读取文件: {path}")
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


# =========================
# Dataset
# =========================

class MustardFullFeatureDataset(Dataset):
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

        self.ids = sorted(list(common_ids))
        self.labels = []

        for sid in self.ids:
            label = self.text[sid]["label"]
            if torch.is_tensor(label):
                label = int(label.item())
            else:
                label = int(label)
            self.labels.append(label)

        print(f"可用样本数: {len(self.ids)}")

        # 打印一个样本看看
        sid = self.ids[0]
        print(f"示例ID: {sid}")
        print("utterance_tokens:", self.text[sid]["utterance_tokens"].shape)
        print("context_tokens:", self.text[sid]["context_tokens"].shape)
        print("utterance_audio:", self.ua[sid]["audio_feature"].shape)
        print("context_audio:", self.ca[sid]["context_audio_feature"].shape)
        print("utterance_video:", self.uv[sid]["video_feature"].shape)
        print("context_video:", self.cv[sid]["context_video_feature"].shape)
        print("label:", self.labels[0])

    def __len__(self):
        return len(self.ids)

    def _get_mask(self, record, key, length):
        if key in record:
            mask = record[key]
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask)
            mask = mask.bool()

            if mask.shape[0] != length:
                mask = torch.ones(length, dtype=torch.bool)

            return mask
        else:
            return torch.ones(length, dtype=torch.bool)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        text_rec = self.text[sid]

        utter_tokens = text_rec["utterance_tokens"].float()
        context_tokens = text_rec["context_tokens"].float()

        utter_text_mask = self._get_mask(
            text_rec,
            "utterance_mask",
            utter_tokens.shape[0]
        )

        context_text_mask = self._get_mask(
            text_rec,
            "context_mask",
            context_tokens.shape[0]
        )

        utter_audio = self.ua[sid]["audio_feature"].float()
        context_audio = self.ca[sid]["context_audio_feature"].float()

        utter_video = self.uv[sid]["video_feature"].float()
        context_video = self.cv[sid]["context_video_feature"].float()

        label = self.labels[idx]

        return {
            "sid": sid,
            "utter_tokens": utter_tokens,
            "context_tokens": context_tokens,
            "utter_text_mask": utter_text_mask,
            "context_text_mask": context_text_mask,
            "utter_audio": utter_audio,
            "context_audio": context_audio,
            "utter_video": utter_video,
            "context_video": context_video,
            "label": torch.tensor(label, dtype=torch.long),
        }


# =========================
# collate：处理变长序列
# =========================

def pad_feature_sequences(seqs):
    """
    输入: list of [L, D]
    输出:
        padded: [B, max_L, D]
        mask:   [B, max_L]  True表示真实位置
    """
    batch_size = len(seqs)
    max_len = max(x.shape[0] for x in seqs)
    dim = seqs[0].shape[1]

    padded = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, x in enumerate(seqs):
        length = x.shape[0]
        padded[i, :length] = x
        mask[i, :length] = True

    return padded, mask


def pad_text_sequences(seqs, masks):
    batch_size = len(seqs)
    max_len = max(x.shape[0] for x in seqs)
    dim = seqs[0].shape[1]

    padded = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
    padded_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (x, m) in enumerate(zip(seqs, masks)):
        length = x.shape[0]
        padded[i, :length] = x

        if m.shape[0] == length:
            padded_mask[i, :length] = m.bool()
        else:
            padded_mask[i, :length] = True

    return padded, padded_mask


def collate_fn(batch):
    utter_tokens, utter_text_mask = pad_text_sequences(
        [x["utter_tokens"] for x in batch],
        [x["utter_text_mask"] for x in batch]
    )

    context_tokens, context_text_mask = pad_text_sequences(
        [x["context_tokens"] for x in batch],
        [x["context_text_mask"] for x in batch]
    )

    utter_audio, utter_audio_mask = pad_feature_sequences(
        [x["utter_audio"] for x in batch]
    )

    context_audio, context_audio_mask = pad_feature_sequences(
        [x["context_audio"] for x in batch]
    )

    utter_video, utter_video_mask = pad_feature_sequences(
        [x["utter_video"] for x in batch]
    )

    context_video, context_video_mask = pad_feature_sequences(
        [x["context_video"] for x in batch]
    )

    labels = torch.stack([x["label"] for x in batch])

    return {
        "utter_tokens": utter_tokens,
        "context_tokens": context_tokens,
        "utter_text_mask": utter_text_mask,
        "context_text_mask": context_text_mask,
        "utter_audio": utter_audio,
        "context_audio": context_audio,
        "utter_video": utter_video,
        "context_video": context_video,
        "utter_audio_mask": utter_audio_mask,
        "context_audio_mask": context_audio_mask,
        "utter_video_mask": utter_video_mask,
        "context_video_mask": context_video_mask,
        "labels": labels,
    }


# =========================
# 工具函数
# =========================

def masked_mean(x, mask):
    """
    x:    [B, L, D]
    mask: [B, L]
    """
    mask = mask.bool()
    mask_float = mask.unsqueeze(-1).float()

    x = x * mask_float
    denom = mask_float.sum(dim=1).clamp(min=1e-6)

    return x.sum(dim=1) / denom


def get_autocast_context(device):
    if device.type == "cuda" and USE_AMP:
        return torch.amp.autocast("cuda")
    else:
        return nullcontext()


# =========================
# 模型
# =========================

class QKVCrossAttentionFullModel(nn.Module):
    def __init__(
        self,
        text_dim=768,
        audio_dim=768,
        video_dim=2048,
        hidden_dim=64,
        num_heads=4,
        dropout=0.4,
        num_classes=2,
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.text_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.text_video_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_text = nn.LayerNorm(hidden_dim)
        self.norm_audio = nn.LayerNorm(hidden_dim)
        self.norm_video = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def cross_attention_pool(
        self,
        text_tokens,
        text_mask,
        other_seq,
        other_mask,
        attn_layer,
    ):
        """
        Q = text_tokens
        K,V = other_seq
        """
        # key_padding_mask: True 表示 padding，所以要取反
        key_padding_mask = ~other_mask.bool()

        attn_out, _ = attn_layer(
            query=text_tokens,
            key=other_seq,
            value=other_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        pooled = masked_mean(attn_out, text_mask)
        return pooled

    def forward(self, batch):
        utter_tokens = batch["utter_tokens"]
        context_tokens = batch["context_tokens"]

        utter_text_mask = batch["utter_text_mask"]
        context_text_mask = batch["context_text_mask"]

        utter_audio = batch["utter_audio"]
        context_audio = batch["context_audio"]

        utter_video = batch["utter_video"]
        context_video = batch["context_video"]

        utter_audio_mask = batch["utter_audio_mask"]
        context_audio_mask = batch["context_audio_mask"]

        utter_video_mask = batch["utter_video_mask"]
        context_video_mask = batch["context_video_mask"]

        # 投影到同一维度
        utter_text = self.norm_text(self.text_proj(utter_tokens))
        context_text = self.norm_text(self.text_proj(context_tokens))

        utter_audio = self.norm_audio(self.audio_proj(utter_audio))
        context_audio = self.norm_audio(self.audio_proj(context_audio))

        utter_video = self.norm_video(self.video_proj(utter_video))
        context_video = self.norm_video(self.video_proj(context_video))

        # text 自身表示
        utter_text_h = masked_mean(utter_text, utter_text_mask)
        context_text_h = masked_mean(context_text, context_text_mask)

        # Q-K-V cross-modal attention
        utter_audio_h = self.cross_attention_pool(
            text_tokens=utter_text,
            text_mask=utter_text_mask,
            other_seq=utter_audio,
            other_mask=utter_audio_mask,
            attn_layer=self.text_audio_attn,
        )

        context_audio_h = self.cross_attention_pool(
            text_tokens=context_text,
            text_mask=context_text_mask,
            other_seq=context_audio,
            other_mask=context_audio_mask,
            attn_layer=self.text_audio_attn,
        )

        utter_video_h = self.cross_attention_pool(
            text_tokens=utter_text,
            text_mask=utter_text_mask,
            other_seq=utter_video,
            other_mask=utter_video_mask,
            attn_layer=self.text_video_attn,
        )

        context_video_h = self.cross_attention_pool(
            text_tokens=context_text,
            text_mask=context_text_mask,
            other_seq=context_video,
            other_mask=context_video_mask,
            attn_layer=self.text_video_attn,
        )

        # 六个表示拼接
        fused = torch.cat(
            [
                utter_text_h,
                context_text_h,
                utter_audio_h,
                context_audio_h,
                utter_video_h,
                context_video_h,
            ],
            dim=1,
        )

        logits = self.classifier(fused)
        return logits


# =========================
# 训练与评价
# =========================

def move_batch_to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with get_autocast_context(device):
            logits = model(batch)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_one_fold(fold_id, train_loader, test_loader, device):
    print("\n" + "=" * 70)
    print(f"Fold {fold_id} / {N_SPLITS}")
    print("=" * 70)

    model = QKVCrossAttentionFullModel(
        text_dim=TEXT_DIM,
        audio_dim=AUDIO_DIM,
        video_dim=VIDEO_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and USE_AMP),
    )

    best_result = {
        "fold": fold_id,
        "epoch": 0,
        "acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    best_model_path = os.path.join(
        SAVE_DIR,
        f"best_qkv_5fold_no_sampling_fold{fold_id}.pt"
    )

    no_improve_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(device):
                logits = model(batch)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            step_count += 1

            if step % 100 == 0:
                print(
                    f"Fold {fold_id} | Epoch {epoch:02d} "
                    f"| Step {step} | Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / max(step_count, 1)

        result = evaluate(model, test_loader, device)

        print(
            f"Fold {fold_id} | Epoch {epoch:02d} "
            f"| Loss: {avg_loss:.4f} "
            f"| Acc: {result['acc']:.4f} "
            f"| P: {result['precision']:.4f} "
            f"| R: {result['recall']:.4f} "
            f"| F1: {result['f1']:.4f}"
        )

        if result["f1"] > best_result["f1"]:
            best_result = {
                "fold": fold_id,
                "epoch": epoch,
                "acc": result["acc"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }

            torch.save(model.state_dict(), best_model_path)
            print(f"Fold {fold_id}: best model saved to {best_model_path}")

            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            print(
                f"Fold {fold_id}: early stopping at epoch {epoch}, "
                f"best epoch = {best_result['epoch']}"
            )
            break

    print(f"\nFold {fold_id} Best result:")
    print(best_result)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_result


# =========================
# 主函数：5-fold cross validation
# =========================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = MustardFullFeatureDataset()

    labels = np.array(dataset.labels)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    all_fold_results = []

    for fold_id, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        train_subset = Subset(dataset, train_idx.tolist())
        test_subset = Subset(dataset, test_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        fold_result = train_one_fold(
            fold_id=fold_id,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        all_fold_results.append(fold_result)

        del train_loader
        del test_loader
        del train_subset
        del test_subset
        torch.cuda.empty_cache()
        gc.collect()

    # =========================
    # 输出最终 5-fold 结果
    # =========================

    print("\n" + "=" * 70)
    print("5-fold cross validation results")
    print("=" * 70)

    for r in all_fold_results:
        print(
            f"Fold {r['fold']} | "
            f"Best Epoch: {r['epoch']} | "
            f"Acc: {r['acc']:.4f} | "
            f"P: {r['precision']:.4f} | "
            f"R: {r['recall']:.4f} | "
            f"F1: {r['f1']:.4f}"
        )

    accs = np.array([r["acc"] for r in all_fold_results])
    ps = np.array([r["precision"] for r in all_fold_results])
    rs = np.array([r["recall"] for r in all_fold_results])
    f1s = np.array([r["f1"] for r in all_fold_results])

    print("\nAverage result:")
    print(f"Accuracy:  {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
    print(f"Precision: {ps.mean():.4f} ± {ps.std(ddof=1):.4f}")
    print(f"Recall:    {rs.mean():.4f} ± {rs.std(ddof=1):.4f}")
    print(f"F1:        {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}")

    csv_path = os.path.join(SAVE_DIR, "qkv_5fold_no_sampling_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["fold", "best_epoch", "accuracy", "precision", "recall", "f1"]
        )

        for r in all_fold_results:
            writer.writerow(
                [
                    r["fold"],
                    r["epoch"],
                    r["acc"],
                    r["precision"],
                    r["recall"],
                    r["f1"],
                ]
            )

        writer.writerow([])
        writer.writerow(
            [
                "mean",
                "",
                accs.mean(),
                ps.mean(),
                rs.mean(),
                f1s.mean(),
            ]
        )
        writer.writerow(
            [
                "std",
                "",
                accs.std(ddof=1),
                ps.std(ddof=1),
                rs.std(ddof=1),
                f1s.std(ddof=1),
            ]
        )

    print(f"\n结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()