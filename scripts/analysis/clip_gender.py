# === clip_gender.py ===
import os
import cv2
import csv
import random
import shutil
import warnings
import time

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    success, frame = cap.read()
    while success:
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
        success, frame = cap.read()
    cap.release()

def ask_clip_gender(image_path, model, processor, device, activity):
    image = Image.open(image_path).convert("RGB")
    prompts = [
        f"a photo of a male person doing {activity}",
        f"a photo of a female person doing {activity}"
    ]
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {
        "pixel_values": image_inputs["pixel_values"].to(device),
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device)
    }
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()
    inference_time = end - start
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    label = "male" if probs.argmax().item() == 0 else "female"
    return label, probs[0].item(), probs[1].item(), inference_time

def process_gender(videos_root, output_dir, tmp_dir, device, sample_size=10, seed=0):
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "clip_summary_gender.csv")
    framewise_csv = os.path.join(output_dir, "clip_framewise_gender.csv")

    completed = set()
    if os.path.exists(summary_csv):
        with open(summary_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                completed.add(row[0])

    fsum = open(summary_csv, "a", newline="")
    fframe = open(framewise_csv, "a", newline="")
    wsum = csv.writer(fsum)
    wframe = csv.writer(fframe)

    if os.stat(summary_csv).st_size == 0:
        wsum.writerow(["video_name", "male_count", "female_count", "final"])
    if os.stat(framewise_csv).st_size == 0:
        wframe.writerow(["video_name", "frame_file", "gender_label", "prob_male", "prob_female", "inference_time_sec"])

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    categories = sorted(os.listdir(videos_root))
    for idx, category in enumerate(categories, 1):
        cat_path = os.path.join(videos_root, category)
        if not os.path.isdir(cat_path):
            continue
        for file in sorted(os.listdir(cat_path)):
            if not file.endswith(".avi"):
                continue
            video_name = file[:-4]
            if video_name in completed:
                continue

            activity = video_name.split("_")[1] if "_" in video_name else "activity"
            video_path = os.path.join(cat_path, file)
            tmp_frames = os.path.join(tmp_dir, video_name)
            extract_frames(video_path, tmp_frames)

            frame_files = sorted(os.listdir(tmp_frames))
            frame_files = [f for f in frame_files if f.endswith(".jpg")]
            if len(frame_files) > sample_size:
                random.seed(seed)
                frame_files = random.sample(frame_files, sample_size)

            male_count = female_count = 0

            for frame in frame_files:
                path = os.path.join(tmp_frames, frame)
                label, pm, pf, infer_time = ask_clip_gender(path, model, processor, device, activity)
                wframe.writerow([video_name, frame, label, f"{pm:.4f}", f"{pf:.4f}", f"{infer_time:.4f}"])
                fframe.flush()
                os.fsync(fframe.fileno())
                if label == "male":
                    male_count += 1
                else:
                    female_count += 1

            final = "male" if male_count > female_count else "female" if female_count > male_count else "uncertain"
            wsum.writerow([video_name, male_count, female_count, final])
            fsum.flush()
            os.fsync(fsum.fileno())
            shutil.rmtree(tmp_frames)
        print(f"[INFO] Completed {idx}/{len(categories)} categories: {category}")

    fsum.close()
    fframe.close()

    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f"[TIMER] Total runtime: {total_minutes:.2f} minutes")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_gender(
        videos_root="/groups/mshah/data/UCF101/videos/",
        output_dir="output_gender/",
        tmp_dir="tmp_frames_gender/",
        device=device,
        sample_size=10,
        seed=0
    )
