import os
import cv2
import csv

def get_video_duration(video_path):
    """Returns the duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None  # If video cannot be opened, return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps > 0:
        return frame_count / fps  # Calculate duration
    return None

def process_videos(folder_path, output_csv):
    """Scans folder for MP4 videos, extracts their names and durations, and saves to CSV."""
    video_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            duration = get_video_duration(video_path)
            if duration is not None:
                video_data.append([filename, round(duration, 2)])

    # Save to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Video Name", "Duration (seconds)"])
        writer.writerows(video_data)

    print(f"CSV file saved: {output_csv}")

# Example usage
folder_path = "./101"  # Update with your folder path
output_csv = "video_durations.csv"
process_videos(folder_path, output_csv)
