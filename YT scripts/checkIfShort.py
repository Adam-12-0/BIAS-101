import os
import pandas as pd
from yt_dlp import YoutubeDL
from tqdm import tqdm

RESULTS_DIR = "youtube_results"
OUTPUT_COLUMN = "Is Portrait"

YDL_OPTS = {
    'quiet': True,
    'skip_download': True,
    'no_warnings': True,
    'extract_flat': False,
    # Removed 'format': 'best' to avoid stream errors
}

def get_video_resolution(url):
    try:
        with YoutubeDL(YDL_OPTS) as ydl:
            info = ydl.extract_info(url, download=False)

            # Some formats may not have width/height, use 'formats' list if needed
            width = info.get("width")
            height = info.get("height")

            # If top-level width/height missing, check in formats
            if not width or not height:
                formats = info.get("formats", [])
                for fmt in reversed(formats):  # Highest quality is usually last
                    if fmt.get("width") and fmt.get("height"):
                        width = fmt["width"]
                        height = fmt["height"]
                        break

            if width and height:
                return height > width
    except Exception as e:
        print(f"Error checking resolution for {url}: {e}")
    return None

def process_csv_file(csv_path):
    df = pd.read_csv(csv_path)

    if OUTPUT_COLUMN in df.columns:
        print(f"Skipping already processed: {csv_path}")
        return

    print(f"Processing {csv_path}...")
    portrait_flags = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking aspect ratios"):
        url = row.get("Link", "")
        is_portrait = get_video_resolution(url)
        portrait_flags.append(is_portrait)

    df[OUTPUT_COLUMN] = portrait_flags
    df.to_csv(csv_path, index=False)
    print(f"✓ Updated {csv_path}")

def main():
    for root, _, files in os.walk(RESULTS_DIR):
        for filename in files:
            if filename.endswith("_results.csv"):
                csv_path = os.path.join(root, filename)
                process_csv_file(csv_path)

if __name__ == "__main__":
    main()
