from collections import Counter
import json
from pathlib import Path

DATA_PATH = Path(r"C:\Users\17962\Desktop\MUStARD-master\data\processed_samples.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

label_counter = Counter([s["label"] for s in samples])
show_counter = Counter([s["show"] for s in samples])

print("label分布:", label_counter)
print("show分布:", show_counter)

print("讽刺样本数:", label_counter[1])
print("非讽刺样本数:", label_counter[0])