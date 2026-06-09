import pickle
from pathlib import Path
import numpy as np
import torch

AUDIO_PATH = Path(r"C:\Users\17962\Desktop\MUStARD-master\data\audio_features.p")

with open(AUDIO_PATH, "rb") as f:
    audio_data = pickle.load(f, encoding="latin1")

print("audio_data 类型:", type(audio_data))

if isinstance(audio_data, dict):
    print("音频样本数:", len(audio_data))

    keys = list(audio_data.keys())
    print("前10个key:")
    for k in keys[:10]:
        print(k)

    first_key = keys[0]
    first_value = audio_data[first_key]

    print("\n第一个key:", first_key)
    print("第一个value类型:", type(first_value))

    if hasattr(first_value, "shape"):
        print("第一个value shape:", first_value.shape)
    elif isinstance(first_value, list):
        print("第一个value是list，长度:", len(first_value))
        if len(first_value) > 0:
            print("list第一个元素类型:", type(first_value[0]))
            if hasattr(first_value[0], "shape"):
                print("list第一个元素shape:", first_value[0].shape)
            else:
                print("list第一个元素内容:", first_value[0])
    elif isinstance(first_value, tuple):
        print("第一个value是tuple，长度:", len(first_value))
        for i, v in enumerate(first_value):
            print(f"tuple[{i}] 类型:", type(v))
            if hasattr(v, "shape"):
                print(f"tuple[{i}] shape:", v.shape)
            else:
                print(f"tuple[{i}] 内容:", v)
    else:
        print("第一个value内容:", first_value)

else:
    print("audio_data 不是 dict")
    print("audio_data 内容类型:", type(audio_data))
    print(audio_data)