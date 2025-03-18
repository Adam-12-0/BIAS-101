import os
import csv
import random
import warnings

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

def ask_clip(image_path, model, processor, device, activity):
    """
    Uses a CLIP model to classify an image as 'young' or 'old' via zero-shot inference,
    incorporating the activity into the prompt.
    """
    # 1. Load and convert the image to RGB
    image = Image.open(image_path).convert("RGB")

    # 2. Define text prompts that include the activity
    text_prompts = [
        f"a photo of a young person doing {activity}",
        f"a photo of an old person doing {activity}"
    ]

    # 3. Preprocess both the image and the text
    inputs = processor(images=image, text=text_prompts, return_tensors="pt", padding=True).to(device)

    # 4. Forward pass through the CLIP model
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Convert logits to probabilities
    logits_per_image = outputs.logits_per_image  # [batch_size, num_text_prompts]
    probs = logits_per_image.softmax(dim=1)[0]
    prob_young = probs[0].item()
    prob_old   = probs[1].item()

    # 6. Determine which text prompt has the highest probability
    label_id = probs.argmax(dim=0).item()
    age_label = "young" if label_id == 0 else "old"

    return age_label, prob_young, prob_old

def process_frames_with_clip(
    frames_root,
    model,
    processor,
    device="cuda",
    sample_size=5,
    seed=42,
    summary_tsv="clip_summary_results.tsv",
    framewise_tsv="clip_framewise_results.tsv"
):
    """
    Iterates over subfolders in frames_root (one subfolder per video),
    extracts the activity name from folder name by splitting on '_',
    randomly samples up to `sample_size` frames, queries CLIP to classify
    each frame as 'young' or 'old' (with the activity in the prompt),
    and writes to two TSV files:
    
    1. framewise_tsv (frame-level):
       [video_name, frame_file, age_label, prob_young, prob_old]
    
    2. summary_tsv (video-level):
       [video_name, young_count, old_count, final]

    'final' is decided via majority vote (young_count vs old_count).
    """
    random.seed(seed)

    with open(summary_tsv, mode="w", newline="", encoding="utf-8") as f_summary, \
         open(framewise_tsv, mode="w", newline="", encoding="utf-8") as f_frames:

        summary_writer = csv.writer(f_summary, delimiter="\t")
        frame_writer = csv.writer(f_frames, delimiter="\t")

        # Write headers
        summary_writer.writerow(["video_name", "young_count", "old_count", "final"])
        frame_writer.writerow(["video_name", "frame_file", "age_label", "prob_young", "prob_old"])

        # Iterate over each video folder
        for video_name in sorted(os.listdir(frames_root)):
            subfolder_path = os.path.join(frames_root, video_name)
            if not os.path.isdir(subfolder_path):
                continue

            # 1) Extract the activity from the folder name by splitting on '_'
            #    We'll assume it's the second element in the split.
            parts = video_name.split("_")
            if len(parts) < 2:
                print(f"Warning: The folder name '{video_name}' doesn't contain an underscore or second element.")
                activity = "activity"
            else:
                activity = parts[1]

            young_count = 0
            old_count   = 0

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
                age_label, prob_young, prob_old = ask_clip(frame_path, model, processor, device, activity)

                # Write frame-level info
                frame_writer.writerow([
                    video_name,
                    frame_file,
                    age_label,
                    f"{prob_young:.4f}",
                    f"{prob_old:.4f}"
                ])

                # Tally the results
                if age_label == "young":
                    young_count += 1
                elif age_label == "old":
                    old_count += 1

            # Decide final label by majority vote
            if young_count > old_count:
                final_label = "young"
            elif old_count > young_count:
                final_label = "old"
            else:
                final_label = "uncertain"

            summary_writer.writerow([video_name, young_count, old_count, final_label])
            print(f"[INFO] {video_name} (Activity: {activity}): young={young_count}, old={old_count}, final={final_label}")


#####################
# Example usage
#####################
if __name__ == "__main__":
    """
    1. Suppose you already extracted frames from each video into:
       frames_root/<video_name>/frame_0.jpg, frame_1.jpg, ...
       where video_name looks like "xxx_activityName_yyy".
    2. This script loads a CLIP model and processor,
       then classifies the sampled frames in each folder,
       using the activity from the folder name in its prompts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    frames_root = "UCF101_extracted_frames/"
    summary_tsv = "clip_summary_results.tsv"
    framewise_tsv = "clip_framewise_results.tsv"

    process_frames_with_clip(
        frames_root=frames_root,
        model=model,
        processor=processor,
        device=device,
        sample_size=5,
        seed=0,
        summary_tsv=summary_tsv,
        framewise_tsv=framewise_tsv
    )
