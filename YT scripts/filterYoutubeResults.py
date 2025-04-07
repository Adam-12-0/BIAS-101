import os
import pandas as pd
from tqdm import tqdm

RESULTS_DIR = "youtube_results"
DURATION_COLUMN = "Duration (seconds)"
PORTRAIT_COLUMN = "Is Portrait"
THUMBNAIL_COLUMN = "Thumbnail Path"

def clean_csv_file(csv_path):
    try:
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if DURATION_COLUMN not in df.columns or PORTRAIT_COLUMN not in df.columns or THUMBNAIL_COLUMN not in df.columns:
            print(f"Skipping {csv_path} — missing required columns.")
            return

        original_count = len(df)

        # Identify rows to remove
        to_remove = df[
            (df[DURATION_COLUMN] < 45) |
            (df[DURATION_COLUMN] > 300) |
            (df[PORTRAIT_COLUMN].astype(bool))
        ]

        # Delete thumbnail files
        if not to_remove.empty:
            csv_dir = os.path.dirname(csv_path)
            for _, row in to_remove.iterrows():
                thumbnail_rel_path = row[THUMBNAIL_COLUMN]
                if pd.notna(thumbnail_rel_path):
                    thumbnail_path = os.path.normpath(os.path.join(csv_dir, thumbnail_rel_path))
                    try:
                        if os.path.exists(thumbnail_path):
                            os.remove(thumbnail_path)
                            print(f"  🗑 Deleted thumbnail: {thumbnail_path}")
                    except Exception as e:
                        print(f"  ⚠️ Failed to delete {thumbnail_path}: {e}")

        # Keep only valid rows
        df_cleaned = df.drop(to_remove.index)
        removed_count = original_count - len(df_cleaned)

        if removed_count > 0:
            df_cleaned.to_csv(csv_path, index=False)
            print(f"✓ Cleaned {csv_path} — removed {removed_count} rows")
        else:
            print(f"No changes for {csv_path}")

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

def main():
    for root, _, files in os.walk(RESULTS_DIR):
        for filename in files:
            if filename.endswith("_results.csv"):
                csv_path = os.path.join(root, filename)
                clean_csv_file(csv_path)

if __name__ == "__main__":
    main()
