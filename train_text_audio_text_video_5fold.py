import os
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

RESULT_CSV = Path(r"E:\mustard\MUStARD-master\new\text_audio_video_ablation_5fold_results.csv")


# =========================================================
# 训练参数：与上一轮 unimodal 实验保持一致
# =========================================================
BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
SEED = 42

MAX_UTTER_TEXT_LEN = 64
MAX_CONTEXT_TEXT_LEN = 256

HIDDEN_DIM = 64
NUM_HEADS = 4
USE_AMP = True

MODEL_NAMES = [
    "text_audio",
    "text_video",
]


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
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pad_seq(seqs):
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
    x:    [batch, seq_len, hidden_dim]
    mask: [batch, seq_len]

    -> [batch, hidden_dim]
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


def make_classifier(input_dim):
    """
    Text + Audio / Text + Video 都会得到4个 hidden_dim 向量：

    utterance text
    context text
    utterance cross-modal
    context cross-modal

    hidden_dim=64 时：
    4 * 64 = 256维输入
    """

    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(64, 2)
    )


# =========================================================
# Dataset
# =========================================================
class AblationDataset(Dataset):

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

        self.labels = [
            int(self.text[sid]["label"])
            for sid in self.ids
        ]

        print("可用样本数:", len(self.ids))
        print("label 0:", self.labels.count(0))
        print("label 1:", self.labels.count(1))

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

        utter_audio = (
            self.ua[sid]["audio_feature"]
            .float()
        )

        context_audio = (
            self.ca[sid]["context_audio_feature"]
            .float()
        )

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

    utter_audio, utter_audio_mask = pad_seq(
        [x["utterance_audio"] for x in batch]
    )

    context_audio, context_audio_mask = pad_seq(
        [x["context_audio"] for x in batch]
    )

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
# Cross-modal 基础模块
# =========================================================
class CrossModalBase(nn.Module):

    def __init__(
        self,
        hidden_dim=64,
        num_heads=4,
        use_audio=True,
        use_video=True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Text: 768 -> H
        self.text_proj = nn.Linear(
            768,
            hidden_dim
        )

        self.text_norm = nn.LayerNorm(
            hidden_dim
        )

        if use_audio:
            # Audio: 768 -> H
            self.audio_proj = nn.Linear(
                768,
                hidden_dim
            )

            self.audio_norm = nn.LayerNorm(
                hidden_dim
            )

            self.text_audio_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            )

        if use_video:
            # Video: 2048 -> H
            self.video_proj = nn.Linear(
                2048,
                hidden_dim
            )

            self.video_norm = nn.LayerNorm(
                hidden_dim
            )

            self.text_video_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
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

    def text_pool(self, tokens, mask):
        """
        Text自身の表現
        [L,768] -> [L,H] -> pooling -> [H]
        """

        h = self.project_text(tokens)

        return masked_mean(
            h,
            mask
        )

    def cross_audio(
        self,
        text_tokens,
        text_mask,
        audio_seq,
        audio_mask
    ):
        """
        Q = Text
        K,V = Audio
        """

        q = self.project_text(
            text_tokens
        )

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

        # attention输出长度与Query(Text)相同
        # [text_len,H] -> [H]
        return masked_mean(
            out,
            text_mask
        )

    def cross_video(
        self,
        text_tokens,
        text_mask,
        video_seq,
        video_mask
    ):
        """
        Q = Text
        K,V = Video
        """

        q = self.project_text(
            text_tokens
        )

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


# =========================================================
# Text + Audio
# =========================================================
class TextAudioModel(CrossModalBase):

    def __init__(
        self,
        hidden_dim=64,
        num_heads=4
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_audio=True,
            use_video=False
        )

        # 4个64维表示 -> 256维
        self.classifier = make_classifier(
            hidden_dim * 4
        )

    def forward(self, batch):

        # Text自身
        utter_text_h = self.text_pool(
            batch["utterance_tokens"],
            batch["utterance_mask"]
        )

        context_text_h = self.text_pool(
            batch["context_tokens"],
            batch["context_mask"]
        )

        # Text -> Audio cross-attention
        utter_audio_h = self.cross_audio(
            batch["utterance_tokens"],
            batch["utterance_mask"],
            batch["utterance_audio"],
            batch["utterance_audio_mask"]
        )

        context_audio_h = self.cross_audio(
            batch["context_tokens"],
            batch["context_mask"],
            batch["context_audio"],
            batch["context_audio_mask"]
        )

        # 4 * 64 = 256
        fused = torch.cat(
            [
                utter_text_h,
                context_text_h,
                utter_audio_h,
                context_audio_h
            ],
            dim=1
        )

        return self.classifier(
            fused
        )


# =========================================================
# Text + Video
# =========================================================
class TextVideoModel(CrossModalBase):

    def __init__(
        self,
        hidden_dim=64,
        num_heads=4
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_audio=False,
            use_video=True
        )

        # 4个64维表示 -> 256维
        self.classifier = make_classifier(
            hidden_dim * 4
        )

    def forward(self, batch):

        # Text自身
        utter_text_h = self.text_pool(
            batch["utterance_tokens"],
            batch["utterance_mask"]
        )

        context_text_h = self.text_pool(
            batch["context_tokens"],
            batch["context_mask"]
        )

        # Text -> Video cross-attention
        utter_video_h = self.cross_video(
            batch["utterance_tokens"],
            batch["utterance_mask"],
            batch["utterance_video"],
            batch["utterance_video_mask"]
        )

        context_video_h = self.cross_video(
            batch["context_tokens"],
            batch["context_mask"],
            batch["context_video"],
            batch["context_video_mask"]
        )

        # 4 * 64 = 256
        fused = torch.cat(
            [
                utter_text_h,
                context_text_h,
                utter_video_h,
                context_video_h
            ],
            dim=1
        )

        return self.classifier(
            fused
        )


def build_model(name):

    if name == "text_audio":
        return TextAudioModel(
            HIDDEN_DIM,
            NUM_HEADS
        )

    if name == "text_video":
        return TextVideoModel(
            HIDDEN_DIM,
            NUM_HEADS
        )

    raise ValueError(
        f"未知模型: {name}"
    )


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
                batch["label"].cpu().tolist()
            )

    total = len(
        all_labels
    )

    correct = sum(
        p == y
        for p, y
        in zip(
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
        for p, y
        in zip(
            all_preds,
            all_labels
        )
    )

    fp = sum(
        p == 1 and y == 0
        for p, y
        in zip(
            all_preds,
            all_labels
        )
    )

    fn = sum(
        p == 0 and y == 1
        for p, y
        in zip(
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
# 单个fold训练
# =========================================================
def train_one_fold(
    model_name,
    fold,
    train_indices,
    test_indices,
    dataset,
    device
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"MODEL = {model_name}"
        f" | FOLD = {fold}"
    )

    print(
        "=" * 70
    )

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
            device.type
            == "cuda"
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        )
    )

    model = build_model(
        model_name
    ).to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            USE_AMP
            and device.type
            == "cuda"
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
                    and device.type
                    == "cuda"
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

        acc, precision, recall, f1 = evaluate(
            model,
            test_loader,
            device
        )

        avg_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            f"Epoch {epoch:02d}"
            f" | Loss {avg_loss:.4f}"
            f" | Acc {acc:.4f}"
            f" | P {precision:.4f}"
            f" | R {recall:.4f}"
            f" | F1 {f1:.4f}"
        )

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
        f" | {model_name}"
        f" | fold {fold}"
        f" | epoch {best['epoch']}"
        f" | Acc {best['acc']:.4f}"
        f" | P {best['precision']:.4f}"
        f" | R {best['recall']:.4f}"
        f" | F1 {best['f1']:.4f}"
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

    dataset = AblationDataset()

    labels = np.array(
        dataset.labels
    )

    # 与上一轮单模态实验完全相同的5-fold生成方式
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    folds = list(
        skf.split(
            np.zeros(
                len(labels)
            ),
            labels
        )
    )

    all_rows = []

    for model_name in MODEL_NAMES:

        results = []

        print(
            "\n\n"
            + "#" * 80
        )

        print(
            "开始模型:",
            model_name
        )

        print(
            "#" * 80
        )

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
                model_name=model_name,
                fold=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                dataset=dataset,
                device=device
            )

            results.append(
                best
            )

            all_rows.append({
                "model": model_name,
                "hidden_dim": HIDDEN_DIM,
                "fold": fold_idx,
                "best_epoch": best["epoch"],
                "accuracy": best["acc"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"]
            })

        accs = np.array([
            x["acc"]
            for x in results
        ])

        ps = np.array([
            x["precision"]
            for x in results
        ])

        rs = np.array([
            x["recall"]
            for x in results
        ])

        f1s = np.array([
            x["f1"]
            for x in results
        ])

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"{model_name}"
            " 5-fold summary"
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
            "-" * 70
        )

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
                "model",
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

    print(
        "\n结果CSV保存到:",
        RESULT_CSV
    )


if __name__ == "__main__":
    main()
