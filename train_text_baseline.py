from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


FEATURE_PATH = Path("text_cls_features.pt")
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextFeatureDataset(Dataset):
    def __init__(self, feature_path):
        data = torch.load(feature_path)

        self.items = []
        for sample_id, item in data.items():
            utterance_cls = item["utterance_cls"]
            context_cls = item["context_cls"]
            label = item["label"]

            # [768] + [768] -> [1536]
            feature = torch.cat([utterance_cls, context_cls], dim=0)

            self.items.append({
                "id": sample_id,
                "feature": feature.float(),
                "label": torch.tensor(label, dtype=torch.long)
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]["feature"], self.items[idx]["label"]


class TextBaselineMLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
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

    # 简单计算 precision / recall / f1 for positive class = sarcasm label 1
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

    dataset = TextFeatureDataset(FEATURE_PATH)
    print("总样本数:", len(dataset))

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

    model = TextBaselineMLP()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"P: {precision:.4f} | "
            f"R: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )


if __name__ == "__main__":
    main()