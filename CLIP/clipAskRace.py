import os
import csv
import random
import warnings

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

def ask_clip_race(image_path, model, processor, device, activity):
    """
    Uses a CLIP model to classify an image by race via zero-shot inference,
    incorporating 'activity' into the prompt.
    """

    # 1. Load and convert the image to RGB
    image = Image.open(image_path).convert("RGB")

    # 2. Define text prompts for each race category, adding the activity
    races = ["white", "asian", "black", "middle eastern", "indian", "mexican"]
    text_prompts = [
        f"a photo of a {race} person doing {activity}" for race in races
    ]

    # 3. Preprocess both the image and the text
    inputs = processor(images=image, text=text_prompts, return_tensors="pt", padding=True).to(device)

    # 4. Forward pass through the CLIP model
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Convert logits to probabilities
    logits_per_image = outputs.logits_per_image  # shape: [1, num_text_prompts]
    probs = logits_per_image.softmax(dim=1)[0]   # shape: [num_text_prompts]

    # 6. Find the race with the highest probability
    label_id = probs.argmax(dim=0).item()
    race_label = races[label_id]

    # Also collect probabilities for each race if needed
    race_probs = [probs[i].item() for i in range(len(races))]

    return race_label, race_probs

def process_frames_with_clip_race(
    frames_root,
    model,
    processor,
    device="cuda",
    sample_size=5,
    seed=42,
    summary_tsv="clip_summary_race.tsv",
    framewise_tsv="clip_framewise_race.tsv"
):
    """
    Iterates over subfolders in frames_root (one subfolder per video),
    extracts the activity name from folder name by splitting on '_',
    randomly samples up to `sample_size` frames, queries CLIP to classify
    each frame by race (with the activity in the prompt),
    and writes results to two TSV files:
    
    1. framewise_tsv (frame-level):
       [video_name, frame_file, race_label, prob_white, prob_asian,
        prob_black, prob_middle_eastern, prob_indian, prob_mexican]
    
    2. summary_tsv (video-level):
       [video_name, count_white, count_asian, count_black,
        count_middle_eastern, count_indian, count_mexican, final_race]

    'final_race' is decided via majority vote across the sampled frames.
    """
    random.seed(seed)

    # Prepare race categories in the same order
    race_categories = ["white", "asian", "black", "middle_eastern", "indian", "mexican"]

    with open(summary_tsv, mode="w", newline="", encoding="utf-8") as f_summary, \
         open(framewise_tsv, mode="w", newline="", encoding="utf-8") as f_frames:

        summary_writer = csv.writer(f_summary, delimiter="\t")
        frame_writer = csv.writer(f_frames, delimiter="\t")

        # Write headers for summary-level results
        summary_writer.writerow([
            "video_name",
            "count_white", "count_asian", "count_black",
            "count_middle_eastern", "count_indian", "count_mexican",
            "final_race"
        ])

        # Write headers for frame-level results
        frame_writer.writerow([
            "video_name",
            "frame_file",
            "race_label",
            "prob_white", "prob_asian", "prob_black",
            "prob_middle_eastern", "prob_indian", "prob_mexican"
        ])

        # Iterate over each video folder
        for video_name in sorted(os.listdir(frames_root)):
            subfolder_path = os.path.join(frames_root, video_name)
            if not os.path.isdir(subfolder_path):
                continue

            # Extract the activity from folder name by splitting on '_'
            parts = video_name.split("_")
            if len(parts) < 2:
                print(f"Warning: The folder name '{video_name}' doesn't contain an underscore or second element.")
                activity = "activity"
            else:
                activity = parts[1]

            # Tally of how many frames are labeled as each race
            race_counts = {
                "white": 0,
                "asian": 0,
                "black": 0,
                "middle_eastern": 0,
                "indian": 0,
                "mexican": 0
            }

            # Gather frame filenames
            frame_files = sorted(
                fn for fn in os.listdir(subfolder_path)
                if fn.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            # Randomly sample up to sample_size frames
            if len(frame_files) > sample_size:
                frame_files = random.sample(frame_files, sample_size)

            # Process each sampled frame
            for frame_file in frame_files:
                frame_path = os.path.join(subfolder_path, frame_file)
                
                # Classify with CLIP, incorporating the activity into the prompt
                race_label, race_probs = ask_clip_race(
                    image_path=frame_path,
                    model=model,
                    processor=processor,
                    device=device,
                    activity=activity
                )

                # Write frame-level info
                frame_writer.writerow([
                    video_name,
                    frame_file,
                    race_label,
                    f"{race_probs[0]:.4f}",
                    f"{race_probs[1]:.4f}",
                    f"{race_probs[2]:.4f}",
                    f"{race_probs[3]:.4f}",
                    f"{race_probs[4]:.4f}",
                    f"{race_probs[5]:.4f}"
                ])

                # Increase the count of the predicted race
                if race_label == "white":
                    race_counts["white"] += 1
                elif race_label == "asian":
                    race_counts["asian"] += 1
                elif race_label == "black":
                    race_counts["black"] += 1
                elif race_label == "middle eastern":
                    race_counts["middle_eastern"] += 1
                elif race_label == "indian":
                    race_counts["indian"] += 1
                elif race_label == "mexican":
                    race_counts["mexican"] += 1

            # Decide final label by majority vote
            final_race = max(race_counts, key=race_counts.get)

            summary_writer.writerow([
                video_name,
                race_counts["white"],
                race_counts["asian"],
                race_counts["black"],
                race_counts["middle_eastern"],
                race_counts["indian"],
                race_counts["mexican"],
                final_race
            ])

            print(
                f"[INFO] {video_name} (Activity: {activity}): {race_counts}, final_race={final_race}"
            )

#####################
# Example usage
#####################
if __name__ == "__main__":
    """
    1. Suppose you already extracted frames from each video into:
       frames_root/<video_name>/frame_0.jpg, frame_1.jpg, ...
       where video_name looks like "xxx_activityName_yyy".
    2. This script loads a CLIP model and processor,
       then classifies the sampled frames in each folder by race
       (with the activity from the folder name in its prompts).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    frames_root = "UCF101_extracted_frames/"
    summary_tsv = "clip_summary_race.tsv"
    framewise_tsv = "clip_framewise_race.tsv"

    process_frames_with_clip_race(
        frames_root=frames_root,
        model=model,
        processor=processor,
        device=device,
        sample_size=5,
        seed=0,
        summary_tsv=summary_tsv,
        framewise_tsv=framewise_tsv
    )
