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
    Uses the LLaVA model to classify an image into one of four age categories:
    'kid', 'teenager', 'middle_aged', or 'old', incorporating the activity into the prompt.
    """
    # 1. Open and preprocess the image
    image = Image.open(image_path)
    processed_image = process_images([image], image_processor, model.config)
    processed_image = [
        _image.to(dtype=torch.float16, device=device) for _image in processed_image
    ]


    # 3. Construct the prompt
    conv_template = "qwen_1_5"
    question = (
        DEFAULT_IMAGE_TOKEN + "\n"
        + "Based on this image, what is the most likely age category of the person doing {Activity}? "
        + "Choose from 'kid', 'teenager', 'middle_aged', or 'old'."
    )
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # 4. Tokenize
    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_sizes = [image.size]

    # 5. Generate output
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


def interpret_age_label(answer):
    """
    Parses LLaVA's response and classifies it into 'kid', 'teenager', 'middle_aged', or 'old'.
    """
    lowered = answer[0].lower()
    if "kid" in lowered:
        return "kid"
    elif "teenager" in lowered:
        return "teenager"
    elif "middle_aged" in lowered:
        return "middle_aged"
    elif "old" in lowered:
        return "old"
    else:
        return None


def process_frames_with_llava(
    frames_root,
    model,
    tokenizer,
    image_processor,
    device="cuda",
    sample_size=5,
    seed=42,
    summary_tsv="age_summary_results.tsv",
    framewise_tsv="age_framewise_results.tsv"
):
    """
    Iterates over subfolders in frames_root (one subfolder per video),
    extracts the activity name from the folder name,
    randomly samples up to `sample_size` frames, queries LLaVA to classify
    each frame into one of four age categories, and writes to two TSV files:

    1. framewise_tsv (frame-level):
       [video_name, frame_file, age_label]

    2. summary_tsv (video-level):
       [video_name, kid_count, teenager_count, middle_aged_count, old_count, final_label]

    'final_label' is determined by majority vote.
    """
    random.seed(seed)

    with open(summary_tsv, mode="w", newline="", encoding="utf-8") as f_summary, \
         open(framewise_tsv, mode="w", newline="", encoding="utf-8") as f_frames:

        summary_writer = csv.writer(f_summary, delimiter="\t")
        frame_writer = csv.writer(f_frames, delimiter="\t")

        # Write headers
        summary_writer.writerow(["video_name", "kid_count", "teenager_count", "middle_aged_count", "old_count", "final"])
        frame_writer.writerow(["video_name", "frame_file", "age_label"])

        # Iterate over each video folder
        for video_name in sorted(os.listdir(frames_root)):
            subfolder_path = os.path.join(frames_root, video_name)
            if not os.path.isdir(subfolder_path):
                continue

            # Extract the activity from the folder name (assuming it's the second element)
            parts = video_name.split("_")
            if len(parts) < 2:
                print(f"Warning: The folder name '{video_name}' doesn't contain an underscore or second element.")
                activity = "activity"
            else:
                activity = parts[1]

            # Initialize counters for each category
            kid_count = 0
            teenager_count = 0
            middle_aged_count = 0
            old_count = 0

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

                # Ask LLaVA with the activity in the prompt
                answer = ask_llava(frame_path, model, tokenizer, image_processor, device, activity)
                age_label = interpret_age_label(answer)

                # Write frame-level info
                frame_writer.writerow([video_name, frame_file, age_label if age_label else "None"])

                # Tally the results
                if age_label == "kid":
                    kid_count += 1
                elif age_label == "teenager":
                    teenager_count += 1
                elif age_label == "middle_aged":
                    middle_aged_count += 1
                elif age_label == "old":
                    old_count += 1

            # Majority vote for final label
            counts = [kid_count, teenager_count, middle_aged_count, old_count]
            labels = ["kid", "teenager", "middle_aged", "old"]
            final_label = labels[counts.index(max(counts))] if max(counts) > 0 else "uncertain"

            # Write video-level summary
            summary_writer.writerow([video_name, kid_count, teenager_count, middle_aged_count, old_count, final_label])
            print(f"[INFO] {video_name} (Activity: {activity}): kid={kid_count}, teenager={teenager_count}, middle_aged={middle_aged_count}, old={old_count}, final={final_label}")


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
    summary_tsv = "workspace/hug/tp/llava_age_summary_results.tsv"
    framewise_tsv = "workspace/hug/tp/llava_age_framewise_results.tsv"

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
