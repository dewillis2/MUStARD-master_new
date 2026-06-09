from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


TEXT_PATH = Path(r"E:\mustard\MUStARD-master\new\text_cls_features.pt")
UTTER_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_audio_wav2vec.pt")
CONTEXT_AUDIO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_audio_wav2vec.pt")
UTTER_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\utterance_video_resnet.pt")
CONTEXT_VIDEO_PATH = Path(r"E:\mustard\MUStARD-master\new\context_video_resnet.pt")

SAVE_PATH = Path(r"E:\mustard\MUStARD-master\new\best_attention_baseline.pt")

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AttentionFeatureDataset(Dataset):
    def __init__(
        self,
        text_path,
        utter_audio_path,
        context_audio_path,
        utter_video_path,
        context_video_path
    ):
        text_data = torch.load(text_path)
        utter_audio_data = torch.load(utter_audio_path)
        context_audio_data = torch.load(context_audio_path)
        utter_video_data = torch.load(utter_video_path)
        context_video_data = torch.load(context_video_path)

        common_ids = (
            set(text_data.keys())
            & set(utter_audio_data.keys())
            & set(context_audio_data.keys())
            & set(utter_video_data.keys())
            & set(context_video_data.keys())
        )

        self.items = []

        for sample_id in sorted(common_ids):
            text_item = text_data[sample_id]

            # text CLS features
            utterance_text = text_item["utterance_cls"].float()  # [768]
            context_text = text_item["context_cls"].float()      # [768]

            # audio sequence features -> mean pooling
            utterance_audio = utter_audio_data[sample_id]["audio_feature"].float().mean(dim=0)  # [768]
            context_audio = context_audio_data[sample_id]["context_audio_feature"].float().mean(dim=0)  # [768]

            # video sequence features -> mean pooling
            utterance_video = utter_video_data[sample_id]["video_feature"].float().mean(dim=0)  # [2048]
            context_video = context_video_data[sample_id]["context_video_feature"].float().mean(dim=0)  # [2048]

            label = torch.tensor(text_item["label"], dtype=torch.long)

            self.items.append({
                "id": sample_id,
                "utterance_text": utterance_text,
                "context_text": context_text,
                "utterance_audio": utterance_audio,
                "context_audio": context_audio,
                "utterance_video": utterance_video,
                "context_video": context_video,
                "label": label
            })

        print("可用样本数:", len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        return {
            "utterance_text": item["utterance_text"],
            "context_text": item["context_text"],
            "utterance_audio": item["utterance_audio"],
            "context_audio": item["context_audio"],
            "utterance_video": item["utterance_video"],
            "context_video": item["context_video"],
            "label": item["label"]
        }


class ModalityAttentionModel(nn.Module):
    """
    第一版 attention baseline:
    先把 text/audio/video/context 的不同特征都投影到同一维度，
    然后用 attention 学习每个模态表示的重要性。
    """

    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()

        # 将不同维度的特征统一投影到 hidden_dim
        self.utterance_text_proj = nn.Linear(768, hidden_dim)
        self.context_text_proj = nn.Linear(768, hidden_dim)

        self.utterance_audio_proj = nn.Linear(768, hidden_dim)
        self.context_audio_proj = nn.Linear(768, hidden_dim)

        self.utterance_video_proj = nn.Linear(2048, hidden_dim)
        self.context_video_proj = nn.Linear(2048, hidden_dim)

        # attention score 层
        # 输入每个模态的 hidden 表示，输出一个重要性分数
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        utterance_text,
        context_text,
        utterance_audio,
        context_audio,
        utterance_video,
        context_video
    ):
        # 每个模态先映射到同一维度
        ut = self.utterance_text_proj(utterance_text)
        ct = self.context_text_proj(context_text)

        ua = self.utterance_audio_proj(utterance_audio)
        ca = self.context_audio_proj(context_audio)

        uv = self.utterance_video_proj(utterance_video)
        cv = self.context_video_proj(context_video)

        # [batch, 6, hidden_dim]
        # 6 个表示分别是:
        # utterance text, context text, utterance audio, context audio, utterance video, context video
        modality_stack = torch.stack([ut, ct, ua, ca, uv, cv], dim=1)

        # [batch, 6, 1]
        scores = self.attention_score(modality_stack)

        # [batch, 6, 1]
        attention_weights = torch.softmax(scores, dim=1)

        # 加权求和: [batch, hidden_dim]
        fused = torch.sum(attention_weights * modality_stack, dim=1)

        logits = self.classifier(fused)

        return logits, attention_weights.squeeze(-1)


def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    correct = 0

    all_preds = []
    all_labels = []
    all_attention_weights = []

    with torch.no_grad():
        for batch in dataloader:
            utterance_text = batch["utterance_text"].to(device)
            context_text = batch["context_text"].to(device)
            utterance_audio = batch["utterance_audio"].to(device)
            context_audio = batch["context_audio"].to(device)
            utterance_video = batch["utterance_video"].to(device)
            context_video = batch["context_video"].to(device)
            labels = batch["label"].to(device)

            logits, attention_weights = model(
                utterance_text,
                context_text,
                utterance_audio,
                context_audio,
                utterance_video,
                context_video
            )

            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_attention_weights.append(attention_weights.cpu())

    acc = correct / total if total > 0 else 0

    tp = sum((p == 1 and y == 1) for p, y in zip(all_preds, all_labels))
    fp = sum((p == 1 and y == 0) for p, y in zip(all_preds, all_labels))
    fn = sum((p == 0 and y == 1) for p, y in zip(all_preds, all_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if len(all_attention_weights) > 0:
        attention_tensor = torch.cat(all_attention_weights, dim=0)
        avg_attention = attention_tensor.mean(dim=0)
    else:
        avg_attention = None

    return acc, precision, recall, f1, avg_attention


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = AttentionFeatureDataset(
        TEXT_PATH,
        UTTER_AUDIO_PATH,
        CONTEXT_AUDIO_PATH,
        UTTER_VIDEO_PATH,
        CONTEXT_VIDEO_PATH
    )

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
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ModalityAttentionModel(hidden_dim=256)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0
    best_result = None

    modality_names = [
        "utterance_text",
        "context_text",
        "utterance_audio",
        "context_audio",
        "utterance_video",
        "context_video"
    ]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            utterance_text = batch["utterance_text"].to(device)
            context_text = batch["context_text"].to(device)
            utterance_audio = batch["utterance_audio"].to(device)
            context_audio = batch["context_audio"].to(device)
            utterance_video = batch["utterance_video"].to(device)
            context_video = batch["context_video"].to(device)
            labels = batch["label"].to(device)

            logits, attention_weights = model(
                utterance_text,
                context_text,
                utterance_audio,
                context_audio,
                utterance_video,
                context_video
            )

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        acc, precision, recall, f1, avg_attention = evaluate(model, test_loader, device)

        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                "epoch": epoch,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_attention": avg_attention
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

        if avg_attention is not None:
            attn_str = " | ".join(
                [f"{name}: {weight:.3f}" for name, weight in zip(modality_names, avg_attention.tolist())]
            )
            print("Avg attention:", attn_str)

    print("\nBest result:")
    print({
        "epoch": best_result["epoch"],
        "acc": best_result["acc"],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1": best_result["f1"]
    })

    print("\nBest average attention weights:")
    best_attn = best_result["avg_attention"]
    for name, weight in zip(modality_names, best_attn.tolist()):
        print(f"{name}: {weight:.4f}")


if __name__ == "__main__":
    main()