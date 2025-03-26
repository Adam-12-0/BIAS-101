import os
import csv
import random
import torch
import copy
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
)
from llava.conversation import conv_templates

import warnings
warnings.filterwarnings("ignore")

##############################
# LLaVA integration
##############################
def ask_llava(image_path, model, tokenizer, image_processor, device, activity):
    """
    Uses the LLaVA model to classify an image's race among:
    'white', 'asian', 'black', 'middle_eastern', 'indian', 'latino'.
    """
    # 1. Open and preprocess the image
    image = Image.open(image_path)
    processed_image = process_images([image], image_processor, model.config)
    processed_image = [
        _image.to(dtype=torch.float16, device=device) for _image in processed_image
    ]

    # 2. Define the question for race detection
    conv_template = "qwen_1_5"
    question = (
        DEFAULT_IMAGE_TOKEN + "\n"
        + "Based on this image, what is the most likely race of the person doing {activity} "
        + "Choose from 'white', 'asian', 'black', 'middle_eastern', 'indian', or 'latino'."
    )
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # 3. Tokenize
    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_sizes = [image.size]

    # 4. Generate output
    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=processed_image,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return answer


def interpret_race(answer):
    """
    Parses LLaVA's response and classifies it into one of the six race categories
    or returns 'uncertain' if not found.
    """
    lowered = answer[0].lower()
    for race in ["white", "asian", "black", "middle_eastern", "indian", "latino"]:
        if race in lowered:
            return race
    return "uncertain"  # Default if race is unclear


def process_frames_with_llava(
    frames_root,
    model,
    tokenizer,
    image_processor,
    device="cuda",
    sample_size=5,
    seed=42,
    summary_tsv="race_summary.tsv",
    framewise_tsv="race_framewise.tsv"
):
    """
    Iterates over subfolders in frames_root, extracts the activity name from folder name,
    randomly samples up to `sample_size` frames, queries LLaVA to classify race,
    and writes to two TSV files:

    1. framewise_tsv (frame-level):
       [video_name, frame_file, race]

    2. summary_tsv (video-level):
       [video_name, white_count, asian_count, black_count, middle_eastern_count, indian_count, latino_count, final_label]
    """
    random.seed(seed)

    with open(summary_tsv, mode="w", newline="", encoding="utf-8") as f_summary, \
         open(framewise_tsv, mode="w", newline="", encoding="utf-8") as f_frames:

        summary_writer = csv.writer(f_summary, delimiter="\t")
        frame_writer = csv.writer(f_frames, delimiter="\t")

        # Write headers
        summary_writer.writerow([
            "video_name",
            "white_count",
            "asian_count",
            "black_count",
            "middle_eastern_count",
            "indian_count",
            "latino_count",
            "final"
        ])
        frame_writer.writerow(["video_name", "frame_file", "race"])

        # Iterate over each video folder
        for video_name in sorted(os.listdir(frames_root)):
            subfolder_path = os.path.join(frames_root, video_name)
            if not os.path.isdir(subfolder_path):
                continue

            # Extract the activity from the folder name
            parts = video_name.split("_")
            if len(parts) < 2:
                print(f"Warning: The folder name '{video_name}' doesn't contain an underscore or second element.")
                activity = "activity"
            else:
                activity = parts[1]

            # Initialize counters for each race
            race_counts = {
                "white": 0,
                "asian": 0,
                "black": 0,
                "middle_eastern": 0,
                "indian": 0,
                "latino": 0
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

                # Ask LLaVA for race classification
                answer = ask_llava(frame_path, model, tokenizer, image_processor, device, activity)
                race = interpret_race(answer)

                # Write frame-level info
                frame_writer.writerow([video_name, frame_file, race])

                # Tally the results
                if race in race_counts:
                    race_counts[race] += 1

            # Determine final label by majority vote
            final_label = max(race_counts, key=race_counts.get) if max(race_counts.values()) > 0 else "uncertain"

            # Write video-level summary
            summary_writer.writerow(
                [video_name]
                + [race_counts["white"], race_counts["asian"], race_counts["black"],
                   race_counts["middle_eastern"], race_counts["indian"], race_counts["latino"]]
                + [final_label]
            )
            print(f"[INFO] {video_name} (Activity: {activity}): {race_counts}, final={final_label}")


#####################
# Example usage
#####################
if __name__ == "__main__":
    """
    1. Suppose you already extracted frames from each video into:
       frames_root/<video_name>/frame_0.jpg, frame_1.jpg, ...
       where video_name looks like "xxx_activityName_yyy".
    2. Load your LLaVA model, tokenizer, and image processor.
    """
    pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    model_name = "llava_qwen"
    device = "cuda:0"

    # Load the LLaVA model (adjust as needed)
    tokenizer, llava_model, image_processor, max_length = load_pretrained_model(
        pretrained,
        None,
        model_name,
        attn_implementation="eager",
        device_map=device
    )
    llava_model.eval()

    frames_root = "workspace/UCF101_extracted_frames/"
    summary_tsv = "workspace/hug/tp/llava_race_summary.tsv"
    framewise_tsv = "workspace/hug/tp/llava_race_framewise.tsv"

    process_frames_with_llava(
        frames_root,
        llava_model,
        tokenizer,
        image_processor,
        device=device,
        sample_size=5,
        seed=0,
        summary_tsv=summary_tsv,
        framewise_tsv=framewise_tsv
    )
