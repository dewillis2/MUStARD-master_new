import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


DATA_PATH = Path(r"C:\Users\17962\Desktop\MUStARD-master\data\processed_samples.json")
OUTPUT_PATH = Path("text_cls_features.pt")

MODEL_NAME = "bert-base-uncased"


def encode_text(text, tokenizer, model, device, max_length):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    # [CLS] feature: [1, 768] -> [768]
    cls_feature = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()

    return cls_feature


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    feature_dict = {}

    with torch.no_grad():
        for sample in tqdm(samples):
            sample_id = sample["id"]
            utterance_text = sample["utterance_text"]
            context_text = sample["context_text"]
            label = sample["label"]

            utterance_cls = encode_text(
                utterance_text,
                tokenizer,
                model,
                device,
                max_length=128
            )

            context_cls = encode_text(
                context_text,
                tokenizer,
                model,
                device,
                max_length=256
            )

            feature_dict[sample_id] = {
                "utterance_cls": utterance_cls,
                "context_cls": context_cls,
                "label": label
            }

    torch.save(feature_dict, OUTPUT_PATH)

    print("保存完成:", OUTPUT_PATH)
    print("样本数:", len(feature_dict))

    first_id = list(feature_dict.keys())[0]
    print("第一个样本ID:", first_id)
    print("utterance_cls shape:", feature_dict[first_id]["utterance_cls"].shape)
    print("context_cls shape:", feature_dict[first_id]["context_cls"].shape)
    print("label:", feature_dict[first_id]["label"])


if __name__ == "__main__":
    main()