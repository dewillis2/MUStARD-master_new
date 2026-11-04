import os
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache"
os.environ["TORCH_HOME"] = r"E:\torch_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path
import random, gc, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold

TEXT_PATH = Path(r"E:\mustard\MUStARD-master\new\text_token_features.pt")
UTTER_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_audio_wav2vec.pt")
CONTEXT_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_audio_wav2vec.pt")
UTTER_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_video_resnet.pt")
CONTEXT_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_video_resnet.pt")

RESULT_CSV = Path(r"E:\mustard\MUStARD-master\new\unimodal_5fold_results.csv")

BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
SEED = 42
HIDDEN_DIM = 64
USE_AMP = True

MAX_UTTER_TEXT_LEN = 64
MAX_CONTEXT_TEXT_LEN = 256

MODEL_NAMES = ["text_only", "audio_only", "video_only"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    print("读取:", path)
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pad_seq(seqs):
    lengths = torch.tensor([x.shape[0] for x in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    max_len = padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


def move_batch_to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def make_classifier(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),
    )


class DatasetAll(Dataset):
    def __init__(self):
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
        self.labels = [int(self.text[sid]["label"]) for sid in self.ids]

        print("样本数:", len(self.ids))
        print("label0:", self.labels.count(0), "label1:", self.labels.count(1))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        return {
            "utterance_tokens": self.text[sid]["utterance_tokens"][:MAX_UTTER_TEXT_LEN].float(),
            "utterance_mask": self.text[sid]["utterance_mask"][:MAX_UTTER_TEXT_LEN].long(),
            "context_tokens": self.text[sid]["context_tokens"][:MAX_CONTEXT_TEXT_LEN].float(),
            "context_mask": self.text[sid]["context_mask"][:MAX_CONTEXT_TEXT_LEN].long(),

            "utterance_audio": self.ua[sid]["audio_feature"].float(),
            "context_audio": self.ca[sid]["context_audio_feature"].float(),

            "utterance_video": self.uv[sid]["video_feature"].float(),
            "context_video": self.cv[sid]["context_video_feature"].float(),

            "label": torch.tensor(self.text[sid]["label"], dtype=torch.long),
        }


def collate_fn(batch):
    utter_tokens = torch.stack([x["utterance_tokens"] for x in batch])
    utter_mask = torch.stack([x["utterance_mask"] for x in batch]).bool()

    context_tokens = torch.stack([x["context_tokens"] for x in batch])
    context_mask = torch.stack([x["context_mask"] for x in batch]).bool()

    utter_audio, utter_audio_mask = pad_seq([x["utterance_audio"] for x in batch])
    context_audio, context_audio_mask = pad_seq([x["context_audio"] for x in batch])

    utter_video, utter_video_mask = pad_seq([x["utterance_video"] for x in batch])
    context_video, context_video_mask = pad_seq([x["context_video"] for x in batch])

    labels = torch.stack([x["label"] for x in batch])

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

        "label": labels,
    }


class TextOnly(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(768, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = make_classifier(hidden_dim * 2)

    def encode(self, x, mask):
        x = self.norm(self.proj(x))
        return masked_mean(x, mask)

    def forward(self, batch):
        u = self.encode(batch["utterance_tokens"], batch["utterance_mask"])
        c = self.encode(batch["context_tokens"], batch["context_mask"])
        return self.classifier(torch.cat([u, c], dim=1))


class AudioOnly(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(768, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = make_classifier(hidden_dim * 2)

    def encode(self, x, mask):
        x = self.norm(self.proj(x))
        return masked_mean(x, mask)

    def forward(self, batch):
        u = self.encode(batch["utterance_audio"], batch["utterance_audio_mask"])
        c = self.encode(batch["context_audio"], batch["context_audio_mask"])
        return self.classifier(torch.cat([u, c], dim=1))


class VideoOnly(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(2048, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = make_classifier(hidden_dim * 2)

    def encode(self, x, mask):
        x = self.norm(self.proj(x))
        return masked_mean(x, mask)

    def forward(self, batch):
        u = self.encode(batch["utterance_video"], batch["utterance_video_mask"])
        c = self.encode(batch["context_video"], batch["context_video_mask"])
        return self.classifier(torch.cat([u, c], dim=1))


def build_model(name):
    if name == "text_only":
        return TextOnly(HIDDEN_DIM)
    if name == "audio_only":
        return AudioOnly(HIDDEN_DIM)
    if name == "video_only":
        return VideoOnly(HIDDEN_DIM)
    raise ValueError(name)


def evaluate(model, loader, device):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(
                enabled=(USE_AMP and device.type == "cuda")
            ):
                logits = model(batch)

            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().tolist())
            labels_all.extend(batch["label"].cpu().tolist())

    n = len(labels_all)
    acc = sum(p == y for p, y in zip(preds, labels_all)) / n

    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels_all))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels_all))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels_all))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return acc, precision, recall, f1


def train_fold(model_name, fold, train_idx, test_idx, dataset, device):
    set_seed(SEED + fold)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(USE_AMP and device.type == "cuda")
    )

    best = {"epoch": 0, "acc": 0, "precision": 0, "recall": 0, "f1": -1}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=(USE_AMP and device.type == "cuda")
            ):
                logits = model(batch)
                loss = criterion(logits, batch["label"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        acc, precision, recall, f1 = evaluate(model, test_loader, device)

        print(
            f"{model_name} | Fold {fold} | Epoch {epoch:02d} | "
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Acc {acc:.4f} | P {precision:.4f} | "
            f"R {recall:.4f} | F1 {f1:.4f}"
        )

        if f1 > best["f1"]:
            best = {
                "epoch": epoch,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    del model, optimizer, scaler, train_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return best


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = DatasetAll()
    labels = np.array(dataset.labels)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    folds = list(skf.split(np.zeros(len(labels)), labels))

    rows = []

    for model_name in MODEL_NAMES:
        results = []

        print("\n" + "=" * 70)
        print("开始:", model_name)
        print("=" * 70)

        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            best = train_fold(
                model_name, fold, train_idx, test_idx, dataset, device
            )
            results.append(best)

            rows.append({
                "model": model_name,
                "hidden_dim": HIDDEN_DIM,
                "fold": fold,
                "best_epoch": best["epoch"],
                "accuracy": best["acc"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
            })

        accs = np.array([r["acc"] for r in results])
        ps = np.array([r["precision"] for r in results])
        rs = np.array([r["recall"] for r in results])
        f1s = np.array([r["f1"] for r in results])

        print("\n" + "-" * 70)
        print(model_name, "5-fold summary")
        print(f"Accuracy : {accs.mean():.4f} ± {accs.std():.4f}")
        print(f"Precision: {ps.mean():.4f} ± {ps.std():.4f}")
        print(f"Recall   : {rs.mean():.4f} ± {rs.std():.4f}")
        print(f"F1       : {f1s.mean():.4f} ± {f1s.std():.4f}")
        print("-" * 70)

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "hidden_dim", "fold", "best_epoch",
                "accuracy", "precision", "recall", "f1"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n结果已保存:", RESULT_CSV)


if __name__ == "__main__":
    main()
