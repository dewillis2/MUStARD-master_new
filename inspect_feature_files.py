from pathlib import Path

ROOT = Path(r"C:\Users\17962\Desktop\MUStARD-master")

for p in ROOT.rglob("*"):
    if p.suffix.lower() in [".p", ".pkl", ".hdf5", ".h5", ".jsonl", ".npy", ".pt"]:
        print(p)