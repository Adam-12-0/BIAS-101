import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_video_sample_sd():
    ucf101_path = 'C:/Users/AdamB/BIAS-101/UCF-101'  # Using forward slashes
    input_file = 'data/#videos.csv'
    output_folder = 'visualizations/'
    os.makedirs(output_folder, exist_ok=True)

    video_data = []

    # Traverse the UCF-101 directory
    for category in os.listdir(ucf101_path):
        category_path = os.path.join(ucf101_path, category)
        if not os.path.isdir(category_path):
            continue

        group_counts = {}

        for video_name in os.listdir(category_path):
            match = re.search(r'_g(\d+)_', video_name)
            if match:
                group_number = int(match.group(1))
                if group_number not in group_counts:
                    group_counts[group_number] = 0
                group_counts[group_number] += 1

        # Calculate the standard deviation of the sample counts per video
        sample_counts = list(group_counts.values())
        if len(sample_counts) > 1:  # Only calculate std if there are multiple samples
            std_sample_count = pd.Series(sample_counts).std()
        else:
            std_sample_count = 0  # No variation if only one sample exists

        video_data.append([category, std_sample_count])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(video_data, columns=['Action Category', 'Sample Count Standard Deviation'])
    os.makedirs('data', exist_ok=True)
    df.to_csv(input_file, index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(df['Action Category'], df['Sample Count Standard Deviation'], color='blue')

    ax.set_title('Standard Deviation of Sample Counts per Video in UCF-101', fontsize=16)
    ax.set_xlabel('Action Category', fontsize=12)
    ax.set_ylabel('Sample Count Standard Deviation', fontsize=12)
    ax.set_xticks(range(len(df['Action Category'])))
    ax.set_xticklabels(df['Action Category'], rotation=45, ha='right', fontsize=5)

    plt.tight_layout()
    output_path = os.path.join(output_folder, '#videos.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved as: {output_path}")


if __name__ == "__main__":
    plot_video_sample_sd()
