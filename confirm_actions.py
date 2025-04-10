import warnings
warnings.filterwarnings("ignore", message="The following named arguments are not valid for `VideoMAEImageProcessor.preprocess`")

import torch
from transformers import AutoProcessor, AutoModel
import cv2
import numpy as np
import os
import argparse
import re
import time
import csv
import random


def vid_to_np(video_path):
    """
    COnverts an MP4 video to a list of numpy arrays
    
    Keyword arguments:
        video_path (string): strin path to an mp4 file
    
    Return: List of a np array representation of each frame
    """
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read frames and store in a list
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    # Close video file
    cap.release()
    return list(np.stack(frames, axis=0))

def subsample_frames(frames_list):
    """
    Subsample a list of frames to exactly 32 evenly spaced frames.

    Args:
        frames_list: List of np arrays of all the frames.

    Return: 32 of the frames, evenly spaced
    """
    if len(frames_list) == 32:
        return frames_list
    indices = np.linspace(0, len(frames_list) - 1, num=32, dtype=int)
    return [frames_list[i] for i in indices]   

def inference(model, processor, video, action_desc):
    """ 
    Performs inference with the xclip model and returns if the action is being performed or not.
    
    Args:
        Model (Hugging Face model): The XCLIP model.
        processor (Hugging Face processor): The XCLIP Processor.
        video(string): path to video file
        action_desc (list) : A list of strings of action descriptions.
    
    Return: Boolean determining whether the model thinks the action is happening or not. 
    """
    
    # Convert video to list of numpy arrays of each frame
    video_list = vid_to_np(video)
    video_list = subsample_frames(video_list) # Subsample or trim the frames to match the expected number (32)

    # Preprocess the video frames and text
    inputs = processor(text=action_desc, videos=video_list, return_tensors="pt", padding=True)
    inputs.to(model.device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_video = outputs.logits_per_video  # Similarity score between video and text
        probs = logits_per_video.softmax(dim=1)  # Convert logits to probabilities

    return probs

def process_dir(dir, out_file, captionsCSV, processor, model, threshold):
    """
    Processes all the scene clips in a directory with each action in its own directory. 
    Saves the results of XCLIP's confidence that the action is being performed or not.
    
    Assumed Dir Struct:
    Folder
        |
        -- Folder(s)
                |
                -- Action-Scenes-000.mp4
                -- Action-Scenes-001.mp4
                ...
    
    Args:
    dir (string): Directory to be processed
    out_file (string): name of the csv file to write out to
    processor: hugging face XCLIP processor
    model: hugging face XCLIP model
    threshold (float) : float of the confidence level you want the model
                        to have to deem the action is being performed
    """
    captions = {}
    # Open CSV for captions
    with open(captionsCSV, 'r') as caps:
        reader = csv.reader(caps)
        for row in reader:
            captions[row[0]] = [row[1], "A person doing something"]

    # Open csv to write
    with open(out_file, "w", newline="") as out:
        # Create writer and write headers
        writer = csv.writer(out, delimiter=',')
        headers = ["file", "action", "XCLIP comfirm confidence", "XCLIP deny confidence", f"outcome ({threshold*100}%)"]
        writer.writerow(headers)


        # Walk through each action folder action folder
        for i, folder in enumerate(os.walk(dir)):
            if i==0: # skip first
                continue

            # Top level path to action folder
            top_level_dir = os.path.join(os.getcwd(), folder[0])
            
            # Extract the action from file name
            first_file = folder[2][0]
            action = first_file[:-14]
            caption = captions[action]

            # Obtain and write XCLIP output for each video file in action folder
            for j, file in enumerate(folder[2]):
                path = os.path.join(top_level_dir, file)
                probs = inference(model, processor, path, caption)
                
                newRow = [
                    os.path.join(folder[0], file), action, f"{(probs[0][0]*100):0.2f}", f"{(probs[0][1]*100):0.2f}", f"{probs[0][0] >= threshold}"]
                writer.writerow(newRow) # Write results to csv

            print(f"{action} directory processed.")

def main(args):
    # Get Device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load processor and model from hugging face
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-ucf-2-shot")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch16-ucf-2-shot").to(device)
    # Get captions
    captionsCSV = "captions.csv"
    
    # Process the directory
    process_dir(args.dir, args.out, captionsCSV, processor, model, args.threshold)
    print("Script complete.")


if __name__ == "__main__":
    # Arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="The directory of the videos you want to test", default=".")
    parser.add_argument("--out", help="Name of the file you want out to", default=f"actions-out-{time.time()}.csv")
    parser.add_argument("--threshold", help="Confidence threshold for the model", default=0.90, type=float)
    
    args = parser.parse_args()
    main(args)


