import os
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache"
os.environ["TORCH_HOME"] = r"E:\torch_cache"

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DATA_PATH = Path(r"E:\mustard\MUStARD-master\data\processed_samples.json")
OUTPUT_PATH = Path(r"E:\mustard\MUStARD-master\new\text_token_features.pt")

MODEL_NAME = "bert-base-uncased"


def encode_text(text, tokenizer, model, device, max_length):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = outputs.last_hidden_state.squeeze(0).cpu()      # [token_len, 768]
    mask = inputs["attention_mask"].squeeze(0).cpu()         # [token_len]

    return tokens, mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    features = {}

    for sample in tqdm(samples):
        sample_id = sample["id"]

        utter_tokens, utter_mask = encode_text(
            sample["utterance_text"],
            tokenizer,
            model,
            device,
            max_length=64
        )

        context_tokens, context_mask = encode_text(
            sample["context_text"],
            tokenizer,
            model,
            device,
            max_length=256
        )

        features[sample_id] = {
            "utterance_tokens": utter_tokens,
            "utterance_mask": utter_mask,
            "context_tokens": context_tokens,
            "context_mask": context_mask,
            "label": sample["label"]
        }

    torch.save(features, OUTPUT_PATH)

    print("保存完成:", OUTPUT_PATH)
    print("样本数:", len(features))

    first_id = list(features.keys())[0]
    print("示例ID:", first_id)
    print("utterance_tokens:", features[first_id]["utterance_tokens"].shape)
    print("context_tokens:", features[first_id]["context_tokens"].shape)


if __name__ == "__main__":
    main()