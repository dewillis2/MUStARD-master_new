from pathlib import Path

UTTERANCE_DIR = Path(r"C:\Users\17962\Desktop\utterances_final")
CONTEXT_DIR = Path(r"C:\Users\17962\Desktop\context_final")

for name, folder in [("utterance", UTTERANCE_DIR), ("context", CONTEXT_DIR)]:
    print(f"\n{name} 文件夹:")
    print("路径:", folder)
    print("是否存在:", folder.exists())

    video_files = []
    for suffix in [".mp4", ".avi", ".mov", ".mkv"]:
        video_files.extend(list(folder.glob(f"*{suffix}")))

    print("视频数量:", len(video_files))

    print("前20个文件:")
    for f in video_files[:20]:
        print(f.name)