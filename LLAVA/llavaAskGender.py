import os
import csv
from PIL import Image
import torch
import copy
import os

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import random
import warnings
warnings.filterwarnings("ignore")
##############################
# Example LLaVA integration
##############################
def ask_llava(image_path, model, tokenizer, image_processor, device):
    """
    Asks the LLaVA model: "Is the person's gender male or female?"
    Returns a raw textual response from the model.
    
    NOTE:
    - This function is a placeholder! You'll need to adapt it to
      your actual LLaVA code and how it accepts image + text prompts.
    """
    # Load and preprocess image
    image = Image.open(image_path)
    processed_image = process_images([image], image_processor, model.config)
    processed_image = [_image.to(dtype=torch.float16, device=device) for _image in processed_image]


    # Construct the question
    # We request a short answer to reduce extraneous text, but LLaVA may still respond in sentences.
    conv_template = "qwen_1_5" 
    question = DEFAULT_IMAGE_TOKEN + "\n" + "What is the person's gender in this image? Please answer with 'male' or 'female'."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # Tokenize question
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]
    
    cont = model.generate(
        input_ids,
        images=processed_image,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    # Generate output (this will differ depending on your LLaVA code)
    answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return answer

def interpret_gender(answer):
    """
    Naive interpretation: If the model's text includes 'male', we pick male;
    if it includes 'female', we pick female; otherwise return None.
    This is simplistic and may need more robust parsing.
    """
    lowered = answer[0].lower()
    if "male" in lowered and "female" not in lowered:
        return "male"
    elif "female" in lowered and "male" not in lowered:
        return "female"
    elif "male" in lowered and "female" in lowered:
        # The model might mention both. Handle tie or partial results.
        # For example, pick whichever appears first. Or default to None.
        # For demonstration, let's pick whichever appears first in text:
        male_index = lowered.index("male")
        female_index = lowered.index("female")
        return "male" if male_index < female_index else "female"
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
    summary_tsv="gender_results.tsv",
    framewise_tsv="gender_framewise_results.tsv"
):
    """
    Iterates over subfolders of frames_root (one subfolder per video),
    randomly samples up to `sample_size` frames, applies LLaVA to each
    to estimate gender, and writes two TSV files:

    1. `framewise_tsv` (frame-level):
        video_name, frame_file, gender
    2. `summary_tsv` (video-level):
        video_name, male_count, female_count, final

    'final' is decided via a simple majority vote (male_count vs female_count).
    """
    random.seed(seed)

    # Open the two TSV files together
    with open(summary_tsv, mode="w", newline="", encoding="utf-8") as f_summary, \
         open(framewise_tsv, mode="w", newline="", encoding="utf-8") as f_frames:

        summary_writer = csv.writer(f_summary, delimiter="\t")
        frame_writer = csv.writer(f_frames, delimiter="\t")

        # Write headers
        summary_writer.writerow(["video_name", "male_count", "female_count", "final"])
        frame_writer.writerow(["video_name", "frame_file", "gender"])

        # Iterate over each video folder
        for video_name in sorted(os.listdir(frames_root)):
            subfolder_path = os.path.join(frames_root, video_name)
            if not os.path.isdir(subfolder_path):
                continue

            male_count = 0
            female_count = 0

            # Gather frame filenames
            frame_files = sorted(
                fn for fn in os.listdir(subfolder_path)
                if fn.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            # Randomly sample frames (up to 'sample_size')
            if len(frame_files) > sample_size:
                frame_files = random.sample(frame_files, sample_size)

            # Process the sampled frames
            for frame_file in frame_files:
                frame_path = os.path.join(subfolder_path, frame_file)
                
                # Query LLaVA
                answer = ask_llava(frame_path, model, tokenizer, image_processor, device)
                gender = interpret_gender(answer)

                # Record the frame-level classification
                frame_writer.writerow([video_name, frame_file, gender if gender else "None"])

                # Tally votes
                if gender == "male":
                    male_count += 1
                elif gender == "female":
                    female_count += 1

            # Majority vote for final label
            if male_count > female_count:
                final_label = "male"
            elif female_count > male_count:
                final_label = "female"
            else:
                final_label = "uncertain"

            # Write video-level summary
            summary_writer.writerow([video_name, male_count, female_count, final_label])
            print(f"[INFO] {video_name}: male={male_count}, female={female_count}, final={final_label}")

#####################
# Example usage
#####################
if __name__ == "__main__":
    """
    1. Suppose you already extracted frames from each video into:
       frames_root/<video_name>/frame_0.jpg, frame_1.jpg, ...
    2. You have your LLaVA model loaded as `llava_model`, plus 
       `tokenizer`, and `image_processor`.
    """
    
    pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    model_name = "llava_qwen"
    device = "cuda:0"
    attn_implementation = "eager"
    tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation = attn_implementation, device_map=device)  
    llava_model.eval()

    frames_root = "workspace/UCF101_extracted_frames/"   # Folder containing subfolders of frames
    summary_tsv = "workspace/hug/tp/gender_results.tsv"
    framewise_tsv = "workspace/hug/tp/gender_framewise_results.tsv"


    # Run the pipeline
    process_frames_with_llava(
        frames_root,
        llava_model,
        tokenizer,
        image_processor,
        device=device,
        sample_size=5,            # random sample size
        seed=0,                  # for reproducibility
        summary_tsv=summary_tsv,
        framewise_tsv=framewise_tsv
    )
