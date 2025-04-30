import os
import csv
import shutil
from pathlib import Path

CSV_PATH = "intern_output4.csv"
SRC_ROOT = Path("vid")

# Parse CSV and collect relative paths of TRUE-action clips
true_action_clips = set()
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) >= 4:
            full_path = Path(row[0])
            rel_clip_path = Path(*full_path.parts[-5:])  # e.g., FrontCrawl/Female/VideoName/Scene-XXX.mp4
            action_flag = row[2].strip().upper()
            if action_flag == "TRUE":
                true_action_clips.add(rel_clip_path)

# Walk through vid and copy TRUE clips into TRUE/ subfolders
for file_path in SRC_ROOT.rglob("*.mp4"):
    rel_path = Path(*file_path.parts[-5:])
    if rel_path in true_action_clips:
        true_folder = file_path.parent / "TRUE"
        true_folder.mkdir(exist_ok=True)
        dst_path = true_folder / file_path.name
        shutil.copy2(file_path, dst_path)
        print(f"[COPIED] {file_path} → {dst_path}")
