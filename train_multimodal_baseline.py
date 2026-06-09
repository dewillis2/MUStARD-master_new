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

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultimodalFeatureDataset(Dataset):
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

            utterance_cls = text_item["utterance_cls"].float()
            context_cls = text_item["context_cls"].float()

            # wav2vec audio: [audio_len, 768] -> [768]
            utter_audio = utter_audio_data[sample_id]["audio_feature"].float().mean(dim=0)
            context_audio = context_audio_data[sample_id]["context_audio_feature"].float().mean(dim=0)

            # ResNet video: [frame_len, 2048] -> [2048]
            utter_video = utter_video_data[sample_id]["video_feature"].float().mean(dim=0)
            context_video = context_video_data[sample_id]["context_video_feature"].float().mean(dim=0)

            # 拼接：
            # text: 768 + 768
            # audio: 768 + 768
            # video: 2048 + 2048
            # total = 7168
            feature = torch.cat(
                [
                    utterance_cls,
                    context_cls,
                    utter_audio,
                    context_audio,
                    utter_video,
                    context_video
                ],
                dim=0
            )

            label = torch.tensor(text_item["label"], dtype=torch.long)

            self.items.append({
                "id": sample_id,
                "feature": feature,
                "label": label
            })

        print("可用多模态样本数:", len(self.items))

        if len(self.items) > 0:
            print("输入特征维度:", self.items[0]["feature"].shape)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]["feature"], self.items[idx]["label"]


class MultimodalMLP(nn.Module):
    def __init__(self, input_dim=7168, hidden_dim=512, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = correct / total if total > 0 else 0

    tp = sum((p == 1 and y == 1) for p, y in zip(all_preds, all_labels))
    fp = sum((p == 1 and y == 0) for p, y in zip(all_preds, all_labels))
    fn = sum((p == 0 and y == 1) for p, y in zip(all_preds, all_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return acc, precision, recall, f1


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = MultimodalFeatureDataset(
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

    model = MultimodalMLP(input_dim=7168)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0
    best_result = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        acc, precision, recall, f1 = evaluate(model, test_loader, device)

        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                "epoch": epoch,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            torch.save(
                model.state_dict(),
                r"E:\mustard\MUStARD-master\new\best_multimodal_baseline.pt"
            )

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