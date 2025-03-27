import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def plot_video_durations():
    ucf101_path = 'C:/Users/AdamB/BIAS-101/UCF-101'  # Using forward slashes
    input_file = 'data/video_durations.csv'
    output_folder = 'visualizations/'
    os.makedirs(output_folder, exist_ok=True)

    video_data = []

    # Traverse the UCF-101 directory
    for category in os.listdir(ucf101_path):
        category_path = os.path.join(ucf101_path, category)
        if not os.path.isdir(category_path):
            continue

        for video_name in os.listdir(category_path):
            if not video_name.endswith('.avi'):
                continue

            video_path = os.path.join(category_path, video_name)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_name}")
                continue

            # Calculate video duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration = total_frames / fps
            else:
                duration = 0
            cap.release()

            video_data.append([video_name, category, duration])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(video_data, columns=['Video Name', 'Action Category', 'Duration (seconds)'])

    # Group by Action Category and calculate standard deviation for each category
    std_durations = df.groupby('Action Category')['Duration (seconds)'].std().reset_index()
    std_durations.columns = ['Action Category', 'Standard Deviation']

    # Merge the standard deviation results with the original DataFrame
    df = pd.merge(df, std_durations, on='Action Category', how='left')

    # Save the updated DataFrame back to the same CSV file
    os.makedirs('data', exist_ok=True)
    df.to_csv(input_file, index=False)

    # Plotting only the standard deviations of each category
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(std_durations['Action Category'], std_durations['Standard Deviation'], color='blue')

    ax.set_title('Standard Deviation of Video Durations by Action Category', fontsize=16)
    ax.set_xlabel('Action Category', fontsize=12)
    ax.set_ylabel('Standard Deviation (seconds)', fontsize=12)
    ax.set_xticks(range(len(std_durations['Action Category'])))
    ax.set_xticklabels(std_durations['Action Category'], rotation=45, ha='right', fontsize=5)  # Adjusted rotation and fontsize

    plt.tight_layout()
    output_path = os.path.join(output_folder, 'video_durations.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved as: {output_path}")


if __name__ == "__main__":
    plot_video_durations()
