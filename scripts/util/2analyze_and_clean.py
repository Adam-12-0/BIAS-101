import os
from pathlib import Path
from collections import defaultdict

# === CONFIGURATION ===
ROOT = Path("vid2")

def is_empty_dir(path):
    return all(p.is_dir() and is_empty_dir(p) or not p.exists() for p in path.iterdir())

def delete_empty_dirs(root):
    deleted = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        if not any(d.iterdir()):
            d.rmdir()
            deleted += 1
    return deleted

def analyze_dataset(root):
    video_count = 0
    clip_count = 0
    category_counter = defaultdict(int)
    clip_counter = defaultdict(int)

    for category_dir in root.iterdir():
        if not category_dir.is_dir():
            continue
        for bias_dir in category_dir.iterdir():
            if not bias_dir.is_dir():
                continue
            for video_folder in bias_dir.iterdir():
                if not video_folder.is_dir():
                    continue
                clips = list(video_folder.glob("Scene-*.mp4"))
                if clips:
                    video_count += 1
                    clip_count += len(clips)
                    category_key = f"{category_dir.name}/{bias_dir.name}"
                    category_counter[category_key] += 1
                    clip_counter[category_key] += len(clips)

    return {
        "video_count": video_count,
        "clip_count": clip_count,
        "category_counter": category_counter,
        "clip_counter": clip_counter
    }

def print_analysis(stats):
    print("\n📊 === DATASET SUMMARY ===")
    print(f"🎞️  Total videos with clips: {stats['video_count']}")
    print(f"🎬  Total clips (Scene-*.mp4): {stats['clip_count']}")
    
    categories = stats["category_counter"]
    clip_distribution = stats["clip_counter"]

    print(f"📂 Total categories (Action/Bias combinations): {len(categories)}")

    if categories:
        avg_videos = sum(categories.values()) / len(categories)
        avg_clips = stats["clip_count"] / stats["video_count"] if stats["video_count"] else 0
        print(f"📈 Average videos per category: {avg_videos:.2f}")
        print(f"📈 Average clips per video: {avg_clips:.2f}")
        
        print("\n🏆 Top 5 Most Populated Categories (by video count):")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
            print(f"  {cat:<40} → {count} videos, {clip_distribution[cat]} clips")

        print("\n📉 Bottom 5 Least Populated Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1])[:5]:
            print(f"  {cat:<40} → {count} videos, {clip_distribution[cat]} clips")

if __name__ == "__main__":
    removed = delete_empty_dirs(ROOT)
    print(f"\n🧹 Removed {removed} empty folders under: {ROOT}\n")
    stats = analyze_dataset(ROOT)
    print_analysis(stats)
