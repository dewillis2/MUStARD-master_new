import os

# ===== 防止缓存写入 C 盘 =====
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache"
os.environ["TORCH_HOME"] = r"E:\torch_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path
import random
import gc
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold


# =========================================================
# 路径
# =========================================================
TEXT_PATH = Path(r"E:\mustard\MUStARD-master\new\text_token_features.pt")

UTTER_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_audio_wav2vec.pt")
CONTEXT_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_audio_wav2vec.pt")

UTTER_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_video_resnet.pt")
CONTEXT_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_video_resnet.pt")

RESULT_CSV = Path(
    r"E:\mustard\MUStARD-master\new\full_projection_dim_5fold_results.csv"
)


# =========================================================
# 训练参数
# =========================================================
BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
SEED = 42

MAX_UTTER_TEXT_LEN = 64
MAX_CONTEXT_TEXT_LEN = 256

NUM_HEADS = 4
USE_AMP = True

# 老师要求比较的 projection dimension
HIDDEN_DIMS = [64, 128, 256]


# =========================================================
# 工具函数
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    print(f"读取文件: {path}")

    try:
        return torch.load(
            path,
            map_location="cpu",
            mmap=True
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


def pad_seq(seqs):
    """
    seqs:
        list of [seq_len, feature_dim]

    returns:
        padded [batch, max_len, feature_dim]
        mask   [batch, max_len]
    """

    lengths = torch.tensor(
        [x.shape[0] for x in seqs],
        dtype=torch.long
    )

    padded = pad_sequence(
        seqs,
        batch_first=True
    )

    max_len = padded.shape[1]

    mask = (
        torch.arange(max_len).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    return padded, mask


def masked_mean(x, mask):
    """
    x:
        [batch, seq_len, hidden_dim]

    mask:
        [batch, seq_len]

    return:
        [batch, hidden_dim]
    """

    mask = mask.unsqueeze(-1).float()

    return (
        (x * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1e-6)
    )


def move_batch_to_device(batch, device):
    moved = {}

    for k, v in batch.items():

        if torch.is_tensor(v):
            moved[k] = v.to(
                device,
                non_blocking=True
            )

        else:
            moved[k] = v

    return moved


# =========================================================
# Dataset
# =========================================================
class FullDataset(Dataset):

    def __init__(self):

        print("正在读取特征文件...")

        self.text = safe_torch_load(
            TEXT_PATH
        )

        self.ua = safe_torch_load(
            UTTER_AUDIO_PATH
        )

        self.ca = safe_torch_load(
            CONTEXT_AUDIO_PATH
        )

        self.uv = safe_torch_load(
            UTTER_VIDEO_PATH
        )

        self.cv = safe_torch_load(
            CONTEXT_VIDEO_PATH
        )

        common_ids = (
            set(self.text.keys())
            & set(self.ua.keys())
            & set(self.ca.keys())
            & set(self.uv.keys())
            & set(self.cv.keys())
        )

        self.ids = sorted(
            common_ids
        )

        self.labels = [
            int(self.text[sid]["label"])
            for sid in self.ids
        ]

        print(
            "可用样本数:",
            len(self.ids)
        )

        print(
            "label 0:",
            self.labels.count(0)
        )

        print(
            "label 1:",
            self.labels.count(1)
        )

        if self.ids:

            sid = self.ids[0]

            print("\n示例ID:", sid)

            print(
                "utterance_tokens:",
                self.text[sid]["utterance_tokens"].shape
            )

            print(
                "context_tokens:",
                self.text[sid]["context_tokens"].shape
            )

            print(
                "utterance_audio:",
                self.ua[sid]["audio_feature"].shape
            )

            print(
                "context_audio:",
                self.ca[sid]["context_audio_feature"].shape
            )

            print(
                "utterance_video:",
                self.uv[sid]["video_feature"].shape
            )

            print(
                "context_video:",
                self.cv[sid]["context_video_feature"].shape
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        sid = self.ids[idx]

        utter_tokens = (
            self.text[sid]["utterance_tokens"]
            [:MAX_UTTER_TEXT_LEN]
            .float()
        )

        utter_mask = (
            self.text[sid]["utterance_mask"]
            [:MAX_UTTER_TEXT_LEN]
            .long()
        )

        context_tokens = (
            self.text[sid]["context_tokens"]
            [:MAX_CONTEXT_TEXT_LEN]
            .float()
        )

        context_mask = (
            self.text[sid]["context_mask"]
            [:MAX_CONTEXT_TEXT_LEN]
            .long()
        )

        # 完整 audio sequence
        utter_audio = (
            self.ua[sid]["audio_feature"]
            .float()
        )

        context_audio = (
            self.ca[sid]["context_audio_feature"]
            .float()
        )

        # 完整 video sequence
        utter_video = (
            self.uv[sid]["video_feature"]
            .float()
        )

        context_video = (
            self.cv[sid]["context_video_feature"]
            .float()
        )

        label = torch.tensor(
            self.text[sid]["label"],
            dtype=torch.long
        )

        return {
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

    # text
    utter_tokens = torch.stack(
        [x["utterance_tokens"] for x in batch]
    )

    utter_mask = torch.stack(
        [x["utterance_mask"] for x in batch]
    ).bool()

    context_tokens = torch.stack(
        [x["context_tokens"] for x in batch]
    )

    context_mask = torch.stack(
        [x["context_mask"] for x in batch]
    ).bool()

    # audio
    utter_audio, utter_audio_mask = pad_seq(
        [x["utterance_audio"] for x in batch]
    )

    context_audio, context_audio_mask = pad_seq(
        [x["context_audio"] for x in batch]
    )

    # video
    utter_video, utter_video_mask = pad_seq(
        [x["utterance_video"] for x in batch]
    )

    context_video, context_video_mask = pad_seq(
        [x["context_video"] for x in batch]
    )

    labels = torch.stack(
        [x["label"] for x in batch]
    )

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
# Full QKV Model
# =========================================================
class FullQKVModel(nn.Module):
    """
    Full model:

    1. utterance text
    2. context text
    3. utterance text -> audio
    4. context text -> audio
    5. utterance text -> video
    6. context text -> video

    每一个最终表示都是 hidden_dim 维。

    因此：
        classifier input = hidden_dim * 6

    hidden_dim=64:
        6*64=384

    hidden_dim=128:
        6*128=768

    hidden_dim=256:
        6*256=1536
    """

    def __init__(
        self,
        hidden_dim,
        num_heads=4
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} "
                f"不能被 num_heads={num_heads} 整除"
            )

        self.hidden_dim = hidden_dim

        # -----------------------------------------
        # Projection
        # -----------------------------------------
        self.text_proj = nn.Linear(
            768,
            hidden_dim
        )

        self.audio_proj = nn.Linear(
            768,
            hidden_dim
        )

        self.video_proj = nn.Linear(
            2048,
            hidden_dim
        )

        self.text_norm = nn.LayerNorm(
            hidden_dim
        )

        self.audio_norm = nn.LayerNorm(
            hidden_dim
        )

        self.video_norm = nn.LayerNorm(
            hidden_dim
        )

        # -----------------------------------------
        # Cross Attention
        # -----------------------------------------
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

        # -----------------------------------------
        # Classifier
        #
        # 输入维度自动随着 hidden_dim 改变
        #
        # 64  -> 384
        # 128 -> 768
        # 256 -> 1536
        # -----------------------------------------
        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim * 6,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                2
            )
        )

    def project_text(self, x):
        return self.text_norm(
            self.text_proj(x)
        )

    def project_audio(self, x):
        return self.audio_norm(
            self.audio_proj(x)
        )

    def project_video(self, x):
        return self.video_norm(
            self.video_proj(x)
        )

    def cross_attend_audio(
        self,
        text_tokens,
        text_mask,
        audio_seq,
        audio_mask
    ):

        # Q = text
        q = self.project_text(
            text_tokens
        )

        # K,V = audio
        k = self.project_audio(
            audio_seq
        )

        v = k

        out, _ = self.text_audio_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=~audio_mask,
            need_weights=False
        )

        return masked_mean(
            out,
            text_mask
        )

    def cross_attend_video(
        self,
        text_tokens,
        text_mask,
        video_seq,
        video_mask
    ):

        # Q = text
        q = self.project_text(
            text_tokens
        )

        # K,V = video
        k = self.project_video(
            video_seq
        )

        v = k

        out, _ = self.text_video_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=~video_mask,
            need_weights=False
        )

        return masked_mean(
            out,
            text_mask
        )

    def forward(self, batch):

        # -----------------------------------------
        # Text自身
        # -----------------------------------------
        utter_text_h = masked_mean(
            self.project_text(
                batch["utterance_tokens"]
            ),
            batch["utterance_mask"]
        )

        context_text_h = masked_mean(
            self.project_text(
                batch["context_tokens"]
            ),
            batch["context_mask"]
        )

        # -----------------------------------------
        # Text-Audio Cross Attention
        # -----------------------------------------
        utter_audio_h = self.cross_attend_audio(
            batch["utterance_tokens"],
            batch["utterance_mask"],
            batch["utterance_audio"],
            batch["utterance_audio_mask"]
        )

        context_audio_h = self.cross_attend_audio(
            batch["context_tokens"],
            batch["context_mask"],
            batch["context_audio"],
            batch["context_audio_mask"]
        )

        # -----------------------------------------
        # Text-Video Cross Attention
        # -----------------------------------------
        utter_video_h = self.cross_attend_video(
            batch["utterance_tokens"],
            batch["utterance_mask"],
            batch["utterance_video"],
            batch["utterance_video_mask"]
        )

        context_video_h = self.cross_attend_video(
            batch["context_tokens"],
            batch["context_mask"],
            batch["context_video"],
            batch["context_video_mask"]
        )

        # -----------------------------------------
        # 6个 hidden_dim 向量连接
        # -----------------------------------------
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

        logits = self.classifier(
            fused
        )

        return logits


# =========================================================
# Evaluation
# =========================================================
def evaluate(
    model,
    dataloader,
    device
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in dataloader:

            batch = move_batch_to_device(
                batch,
                device
            )

            with torch.cuda.amp.autocast(
                enabled=(
                    USE_AMP
                    and device.type == "cuda"
                )
            ):

                logits = model(
                    batch
                )

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().tolist()
            )

            all_labels.extend(
                batch["label"]
                .cpu()
                .tolist()
            )

    total = len(
        all_labels
    )

    correct = sum(
        p == y
        for p, y in zip(
            all_preds,
            all_labels
        )
    )

    acc = (
        correct / total
        if total
        else 0.0
    )

    tp = sum(
        p == 1 and y == 1
        for p, y in zip(
            all_preds,
            all_labels
        )
    )

    fp = sum(
        p == 1 and y == 0
        for p, y in zip(
            all_preds,
            all_labels
        )
    )

    fn = sum(
        p == 0 and y == 1
        for p, y in zip(
            all_preds,
            all_labels
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    return (
        acc,
        precision,
        recall,
        f1
    )


# =========================================================
# Train one fold
# =========================================================
def train_one_fold(
    hidden_dim,
    fold,
    train_indices,
    test_indices,
    dataset,
    device
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"HIDDEN_DIM = {hidden_dim}"
        f" | FOLD = {fold}"
    )

    print(
        "=" * 80
    )

    # 同一fold下固定随机种子
    set_seed(
        SEED + fold
    )

    train_dataset = Subset(
        dataset,
        train_indices.tolist()
    )

    test_dataset = Subset(
        dataset,
        test_indices.tolist()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    model = FullQKVModel(
        hidden_dim=hidden_dim,
        num_heads=NUM_HEADS
    ).to(
        device
    )

    # 打印一下当前结构
    print(
        f"classifier input dim = "
        f"{hidden_dim * 6}"
    )

    print(
        f"per-head dim = "
        f"{hidden_dim // NUM_HEADS}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            USE_AMP
            and device.type == "cuda"
        )
    )

    best = {
        "epoch": 0,
        "acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0
    }

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            batch = move_batch_to_device(
                batch,
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=(
                    USE_AMP
                    and device.type == "cuda"
                )
            ):

                logits = model(
                    batch
                )

                loss = criterion(
                    logits,
                    batch["label"]
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            total_loss += (
                loss.item()
            )

        avg_loss = (
            total_loss
            / max(
                len(train_loader),
                1
            )
        )

        acc, precision, recall, f1 = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"Epoch {epoch:02d}"
            f" | Loss {avg_loss:.4f}"
            f" | Acc {acc:.4f}"
            f" | P {precision:.4f}"
            f" | R {recall:.4f}"
            f" | F1 {f1:.4f}"
        )

        # 与前面实验保持相同方式：
        # 每一折记录10个epoch中的best F1
        if f1 > best["f1"]:

            best = {
                "epoch": epoch,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        if device.type == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

    print(
        f"\nBEST"
        f" | hidden_dim={hidden_dim}"
        f" | fold={fold}"
        f" | epoch={best['epoch']}"
        f" | Acc={best['acc']:.4f}"
        f" | P={best['precision']:.4f}"
        f" | R={best['recall']:.4f}"
        f" | F1={best['f1']:.4f}"
    )

    del (
        model,
        optimizer,
        scaler,
        train_loader,
        test_loader
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return best


# =========================================================
# Main
# =========================================================
def main():

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "使用设备:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    dataset = FullDataset()

    labels = np.array(
        dataset.labels
    )

    # 与前面实验完全一致
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    # 固定5个fold
    folds = list(
        skf.split(
            np.zeros(
                len(labels)
            ),
            labels
        )
    )

    all_rows = []

    summaries = {}

    # =====================================================
    # 依次跑 64 / 128 / 256
    # =====================================================
    for hidden_dim in HIDDEN_DIMS:

        print(
            "\n\n"
            + "#" * 90
        )

        print(
            f"开始实验:"
            f" hidden_dim = {hidden_dim}"
        )

        print(
            "#"
            * 90
        )

        fold_results = []

        for (
            fold_idx,
            (
                train_idx,
                test_idx
            )
        ) in enumerate(
            folds,
            start=1
        ):

            best = train_one_fold(
                hidden_dim=hidden_dim,
                fold=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                dataset=dataset,
                device=device
            )

            fold_results.append(
                best
            )

            all_rows.append({
                "hidden_dim": hidden_dim,
                "fold": fold_idx,
                "best_epoch": best["epoch"],
                "accuracy": best["acc"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"]
            })

        accs = np.array([
            x["acc"]
            for x in fold_results
        ])

        ps = np.array([
            x["precision"]
            for x in fold_results
        ])

        rs = np.array([
            x["recall"]
            for x in fold_results
        ])

        f1s = np.array([
            x["f1"]
            for x in fold_results
        ])

        summaries[hidden_dim] = {
            "acc_mean": accs.mean(),
            "acc_std": accs.std(),

            "p_mean": ps.mean(),
            "p_std": ps.std(),

            "r_mean": rs.mean(),
            "r_std": rs.std(),

            "f1_mean": f1s.mean(),
            "f1_std": f1s.std()
        }

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"HIDDEN_DIM = {hidden_dim}"
            " | 5-fold summary"
        )

        print(
            f"Accuracy : "
            f"{accs.mean():.4f}"
            f" ± {accs.std():.4f}"
        )

        print(
            f"Precision: "
            f"{ps.mean():.4f}"
            f" ± {ps.std():.4f}"
        )

        print(
            f"Recall   : "
            f"{rs.mean():.4f}"
            f" ± {rs.std():.4f}"
        )

        print(
            f"F1       : "
            f"{f1s.mean():.4f}"
            f" ± {f1s.std():.4f}"
        )

        print(
            "-" * 80
        )

    # =====================================================
    # CSV 保存
    # =====================================================
    RESULT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hidden_dim",
                "fold",
                "best_epoch",
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    # =====================================================
    # 最终对比
    # =====================================================
    print(
        "\n\n"
        + "=" * 90
    )

    print(
        "Projection Dimension 最终比较"
    )

    print(
        "=" * 90
    )

    for hidden_dim in HIDDEN_DIMS:

        s = summaries[
            hidden_dim
        ]

        print(
            f"dim={hidden_dim:3d}"
            f" | classifier input="
            f"{hidden_dim * 6:4d}"
            f" | head dim="
            f"{hidden_dim // NUM_HEADS:3d}"
            f" | Acc "
            f"{s['acc_mean']:.4f}"
            f" ± {s['acc_std']:.4f}"
            f" | P "
            f"{s['p_mean']:.4f}"
            f" ± {s['p_std']:.4f}"
            f" | R "
            f"{s['r_mean']:.4f}"
            f" ± {s['r_std']:.4f}"
            f" | F1 "
            f"{s['f1_mean']:.4f}"
            f" ± {s['f1_std']:.4f}"
        )

    print(
        "\n结果CSV保存到:",
        RESULT_CSV
    )

    print(
        "\n注意：本脚本为了和前面的探索实验保持一致，"
        "每折使用test F1选择best epoch。"
        "正式论文实验建议后续改为validation选择epoch。"
    )


if __name__ == "__main__":
    main()
