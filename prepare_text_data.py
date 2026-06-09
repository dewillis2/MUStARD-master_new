import json
from pathlib import Path

# 这里改成你的实际路径
DATA_PATH = Path(r"C:\Users\17962\Desktop\MUStARD-master\data\sarcasm_data.json")
OUTPUT_PATH = Path(r"C:\Users\17962\Desktop\MUStARD-master\data\processed_samples.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

samples = []

for sample_id, item in raw_data.items():
    utterance_text = item["utterance"]
    context_list = item["context"]
    speaker = item["speaker"]
    context_speakers = item["context_speakers"]
    show = item["show"]

    # context 原来是 list，这里先合成一个字符串，方便后面 BERT 处理
    context_text = " ".join(context_list)

    # sarcasm: true / false → label: 1 / 0
    label = 1 if item["sarcasm"] else 0

    sample = {
        "id": sample_id,
        "utterance_text": utterance_text,
        "context_text": context_text,
        "speaker": speaker,
        "context_speakers": context_speakers,
        "show": show,
        "label": label
    }

    samples.append(sample)

print("样本数量:", len(samples))
print("第一个样本:")
print(samples[0])

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print("保存完成:", OUTPUT_PATH)