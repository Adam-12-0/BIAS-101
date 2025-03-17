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

csv_file = os.path.join(data_folder, "#videos.csv")
output_image = os.path.join(visualization_folder, "#videos.png")

# Ensure folders exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(visualization_folder, exist_ok=True)

# Read the CSV file (Correct delimiter to comma)
df = pd.read_csv(csv_file, delimiter=',')

# Strip whitespace from column names (just in case)
df.columns = df.columns.str.strip()

# Extract category names and video counts
categories = df["Category"]
video_counts = df["# Videos"]

# Compute average number of videos
average_videos = video_counts.mean()

# Create the bar plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(categories, video_counts, color='deepskyblue')

# Add horizontal average line
ax.axhline(y=average_videos, color='red', linestyle='dotted', linewidth=2, label=f'Avg: {average_videos:.2f}')

# Format x-axis labels to be diagonal and smaller
ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=4)

# Labels and title
ax.set_xlabel("Action Category", fontsize=12, color="white")
ax.set_ylabel("Number of Videos", fontsize=12, color="white")
ax.set_title("Number of Videos per Action Category in UCF-101", fontsize=14, color="white")

# Add legend
ax.legend(fontsize=10)

# Ensure layout fits
plt.tight_layout()

# Save the dark theme plot
plt.savefig(output_image, dpi=300)
plt.show()
