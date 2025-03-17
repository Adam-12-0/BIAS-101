import pandas as pd
import matplotlib.pyplot as plt
import os

# Enable dark mode
plt.style.use("dark_background")

# Define base directory (go up one level from 'scripts/')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to base directory
data_folder = os.path.join(base_dir, "data")
visualization_folder = os.path.join(base_dir, "visualizations")

csv_file = os.path.join(data_folder, "video_durations.csv")
output_image = os.path.join(visualization_folder, "video_durations.png")

# Ensure folders exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(visualization_folder, exist_ok=True)

# Read the CSV file (comma-separated)
df = pd.read_csv(csv_file)

# Strip whitespace from column names (just in case)
df.columns = df.columns.str.strip()

# Extract video names and durations
video_names = df["Video Name"]
durations = df["Duration (seconds)"]

# Compute average duration
average_duration = durations.mean()

# Create the bar plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(video_names, durations, color='deepskyblue')

# Add horizontal average line
ax.axhline(y=average_duration, color='red', linestyle='dotted', linewidth=2, label=f'Avg: {average_duration:.2f} sec')

# Format x-axis labels to be diagonal and smaller
ax.set_xticklabels(video_names, rotation=45, ha="right", fontsize=5)

# Labels and title
ax.set_xlabel("Video Name", fontsize=12, color="white")
ax.set_ylabel("Duration (seconds)", fontsize=12, color="white")
ax.set_title("Video Durations in UCF-101", fontsize=14, color="white")

# Add legend
ax.legend(fontsize=10)

# Ensure layout fits
plt.tight_layout()

# Save the dark theme plot
plt.savefig(output_image, dpi=300)
plt.show()
