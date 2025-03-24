import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_video_durations():
    input_file = 'data/video_durations.csv'
    output_folder = 'visualizations/'
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_file)

    if not {'Video Name', 'Duration (seconds)'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'Video Name' and 'Duration (seconds)' columns.")

    average_duration = df['Duration (seconds)'].mean()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(df['Video Name'], df['Duration (seconds)'], color='blue')
    ax.axhline(y=average_duration, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {average_duration:.2f} sec')

    ax.set_title('Video Durations in UCF-101', fontsize=16)
    ax.set_xlabel('Video Name', fontsize=12)
    ax.set_ylabel('Duration (seconds)', fontsize=12)
    ax.set_xticklabels(df['Video Name'], rotation=45, ha='right', fontsize=5)
    ax.legend(loc='upper right', fontsize=15)
    plt.tight_layout()

    output_path = os.path.join(output_folder, 'video_durations.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved as: {output_path}")


if __name__ == "__main__":
    plot_video_durations()
