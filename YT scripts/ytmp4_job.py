import os
import csv
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import detect, AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode

# === CONFIGURATION ===
CSV_FILE = Path("search_results.csv")
OUT_DIR = Path("vid1")
MAX_WORKERS = 30
SPLIT_WORKERS_PER_VIDEO = 8

# === LOGGING ===
def log(msg): print(f"\033[94m[INFO]\033[0m {msg}")
def warn(msg): print(f"\033[91m[WARN]\033[0m {msg}")

# === UTILITIES ===
def sanitize(text):
    return "".join(c for c in text if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")

def get_resume_index(rows):
    for idx, row in enumerate(rows):
        title = sanitize(row["Title"] or row["Video ID"])
        action = sanitize(row["Action"])
        bias = sanitize(row["Bias"])
        folder_path = OUT_DIR / action / bias / title
        scene_files = list(folder_path.glob("Scene-*.mp4"))
        if not scene_files:
            return max(0, idx - 1)
    return 0

def find_scenes(input_path):
    try:
        return detect(input_path, AdaptiveDetector())
    except Exception as e:
        warn(f"Scene detection failed for {input_path}: {e}")
        return []

def split_scenes(input_path, scene_list, output_dir):
    from concurrent.futures import ThreadPoolExecutor

    def split_scene(index, start: FrameTimecode, end: FrameTimecode):
        scene_file = Path(output_dir) / f"Scene-{index:03d}.mp4"
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{start.get_seconds()}",
            "-i", str(input_path),
            "-t", f"{end.get_seconds() - start.get_seconds():.3f}",
            "-c", "copy",
            str(scene_file)
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            warn(f"❌ Failed to split scene {index}: {e}")

    if not scene_list:
        warn("⚠️ No scenes to split.")
        return

    log(f"🧵 Splitting {len(scene_list)} scenes using {SPLIT_WORKERS_PER_VIDEO} threads...")
    with ThreadPoolExecutor(max_workers=SPLIT_WORKERS_PER_VIDEO) as executor:
        futures = [
            executor.submit(split_scene, idx + 1, start, end)
            for idx, (start, end) in enumerate(scene_list)
        ]
        for future in as_completed(futures):
            _ = future.result()

# === DOWNLOAD + SPLIT TASK ===
def process_video(row):
    if row.get("Is_Duplicate", "").strip().lower() == "yes":
        return

    action = row.get("Action", "").strip()
    bias = row.get("Bias", "").strip()
    video_id = row.get("Video ID", "").strip()
    title = row.get("Title", "").strip() or video_id

    if not all([action, bias, video_id]):
        warn(f"❌ Missing metadata in row: {row}")
        return

    title_clean = sanitize(title)
    folder_path = OUT_DIR / sanitize(action) / sanitize(bias) / title_clean
    mp4_path = folder_path / "source.mp4"
    folder_path.mkdir(parents=True, exist_ok=True)

    scene_files = list(folder_path.glob("Scene-*.mp4"))

    if scene_files:
        log(f"⏩ Skipping (already split): {title_clean}")
        return

    if not mp4_path.exists():
        try:
            if not video_id or len(video_id) > 20:
                warn(f"❌ Invalid or missing video ID: {video_id}")
                return

            url = f"https://www.youtube.com/watch?v={video_id}"
            cmd = f'yt-dlp --cookies cookies.txt -o "{mp4_path}" "{url}"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                warn(f"❌ Download failed for video ID: {video_id}")
                warn(f"    stderr: {result.stderr.strip()[:500]}")
                with open("download_errors.log", "a", encoding="utf-8") as errlog:
                    errlog.write(f"[{video_id}] {title_clean}\n")
                    errlog.write(f"COMMAND: {cmd}\n")
                    errlog.write(result.stderr + "\n\n")
                return
            else:
                log(f"✔️ Downloaded: {mp4_path.name}")
        except Exception as e:
            warn(f"❌ Download exception for {video_id}: {e}")
            return

    else:
        log(f"♻️ Found existing source.mp4 for {title_clean}, proceeding to split.")

    scenes = find_scenes(str(mp4_path))
    if scenes:
        split_scenes(str(mp4_path), scenes, folder_path)
        log(f"🎬 Split into {len(scenes)} scenes: {title_clean}")
    else:
        fallback_clip = folder_path / "Scene-000.mp4"
        mp4_path.rename(fallback_clip)
        log(f"📼 Renamed source.mp4 to fallback clip: {fallback_clip.name}")

    if scenes:
        try:
            mp4_path.unlink()
        except Exception as e:
            warn(f"Could not delete source file: {mp4_path}: {e}")
    else:
        log(f"🗂️ Keeping source.mp4 for manual review: {title_clean}")

# === MAIN ENTRY ===
FAILED_VIDEO_IDS = set()

def process_video_with_failure_tracking(row):
    global FAILED_VIDEO_IDS
    video_id = row.get("Video ID", "").strip()
    result = process_video(row)
    if result is False and video_id:
        FAILED_VIDEO_IDS.add(video_id)

def main():
    global FAILED_VIDEO_IDS

    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    start_idx = get_resume_index(rows)
    log(f"📌 Resuming from row {start_idx} of {len(rows)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_video_with_failure_tracking, row) for row in rows[start_idx:]]
        for future in as_completed(futures):
            _ = future.result()

    if FAILED_VIDEO_IDS:
        log(f"🧹 Removing {len(FAILED_VIDEO_IDS)} failed rows from CSV...")
        new_rows = [row for row in rows if row.get("Video ID", "").strip() not in FAILED_VIDEO_IDS]
        with open(CSV_FILE, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_rows[0].keys())
            writer.writeheader()
            writer.writerows(new_rows)
        log("✅ CSV updated to exclude failed videos.")


if __name__ == "__main__":
    main()
