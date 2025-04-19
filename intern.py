import warnings
warnings.filterwarnings("ignore", message="The following named arguments are not valid for `VideoMAEImageProcessor.preprocess`")
warnings.filterwarnings(
    "ignore",
    message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly"
)
warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`.*`top_p` is set to `0\.9`"
)
warnings.filterwarnings(
    "ignore",
    message=r"Setting `pad_token_id` to `eos_token_id`"
)
import os
import argparse
import time
import csv
import re
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, logging
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge("torch")

'''
need to: 
pip install decord flash_attn
pip install transformers==4.39.1
pip install peft==0.5.0
pip install timm easydict einops
pip install sentencepiece
pip install dotenv --> May not be needed depending on how you logon to hugging face.

Agree to: https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B
Agree to: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
'''
from dotenv import load_dotenv



def get_index(num_frames, num_segments):
    '''
    From model docs see for details
    Link: https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B
    '''
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    '''
    From model docs see for details
    Link: https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B
    '''
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
    

def inference_action(model, tokenizer, video_tensor, action):
    """ 
    Performs inference with the InternVideo2-Chat-8B model and returns if the action is being performed or not.
    
    Args:
        Model (Hugging Face model): The InternVideo2 model.
        tokenizer (Hugging Face processor): The InternVideo2 tokenizer.
        video_tensor (pytorch tensor): video tensor created in the inference_contains_person() function
        action (string) : A strings of the action.
    
    Return: Boolean determining whether the model thinks the action is happening or not
    """
    
    # Create chat
    chat = (
        f"Anwser with a yes or no. In this video can you clearly see a person in the middle of performing an action that fits in the category of {action}? "
        f"There will be text relating to the action category ignore that text, Ignore any text in the video. "
        f"If there is not a person in the video, answer with no. "
        f"If the person is only talking about the action category, answer with no. "
        f"If the person is preping to do or about to do the action, BUT does not atually do the action, answer with no. "
        f"If the person completes the ENTIRE action, answer with yes."
    )

    # Run through model 
    chat_history= []
    response, chat_history = model.chat(tokenizer, '', chat, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config={'do_sample':False})

    # Regex to find yes inside the response
    pattern = re.compile(r'\byes\b', re.IGNORECASE) 
    yes_found = bool(pattern.search(response))

    return yes_found


def inference_contains_person(model, tokenizer, video_path):
    """
    Convert the videos to a tensor and runs inference to detect if the video contains a person.
    
    Args:
        Model (Hugging Face model): The InternVideo2 model.
        tokenizer (Hugging Face processor): The InternVideo2 tokenizer.
        video_tensor (pytorch tensor): video tensor created in the inference_contains_person() function
    
    Return: a bool if a person is found and the tensor representation of the video
    """
    # Create chat
    chat = "Anwser with a yes or no. Is there a real person at any point in this video? Again, Answer with a yes or a no."

    # Sample uniformly 8 frames from the video
    video_tensor = load_video(video_path, num_segments=8, return_msg=False)
    video_tensor = video_tensor.to(model.device)

    # Run through model 
    chat_history= []
    response, chat_history = model.chat(tokenizer, '', chat, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config={'do_sample':False})

    pattern = re.compile(r'\byes\b', re.IGNORECASE) # Regex to find yes inside the response
    person_found = bool(pattern.search(response))

    return person_found, video_tensor


def inference_bias(model, tokenizer, video_tensor):
    """
    Runs inference for the 5 bias' types 
    
    Args:
        Model (Hugging Face model): The InternVideo2 model.
        tokenizer (Hugging Face processor): The InternVideo2 tokenizer.
        video_tensor (pytorch tensor): video tensor created in the inference_contains_person() function
    
    Return: a string for each of the 5 bias types
    """
    
    
    chat_history= []
    chat_gender = (
        "Is the person in this video a male or a female? "
        "Please answer with only a 'male' or a 'female' reponse. "
    )
    # Run through model 
    response, chat_history = model.chat(tokenizer, '', chat_gender, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config={'do_sample':False})
    # Regex to find female inside the response
    pattern_gender = re.compile(r'\bfemale\b', re.IGNORECASE) 
    gender = 'female' if bool(pattern_gender.search(response)) else 'male'

    
    chat_race = (
        "Select the race of the person in this video from the races listed.\n "
        "A: Asian\nB: Black\nC: Middle eastern\nD:Indian\nE: White\nF: Hispanic\n"
        "Please only answer with the CAPTIAL letter of your choice. Again, captial letter from the 6 choices. "
    )
    # Run through model 
    response, chat_history = model.chat(tokenizer, '', chat_race, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config={'do_sample':False})
    # Regex to extract 
    m = re.search(r"\b[A-F]\b", response)
    choice = m.group(0) if m else "None"
    letter_to_race = {
        'A': 'asian',
        'B': 'black',
        'C': 'middle eastern',
        'D': 'indian',
        'E': 'white',
        'F': 'hispanic',
    }
    race = letter_to_race.get(choice, "None")


    chat_age = (
        "Select the age of the person in this video from the age categories listed.\n "
        "A: Kid\nB: Young\nC: Old\n"
        "Please only answer with the CAPTIAL letter of your choice. Again, captial letter from the 3 choices. "
    )
     # Run through model 
    response, chat_history = model.chat(tokenizer, '', chat_age, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config={'do_sample':False})
    # Regex to extract 
    m = re.search(r"\b[A-C]\b", response)
    choice = m.group(0) if m else "None"
    letter_to_age = {
        'A': 'kid',
        'B': 'young',
        'C': 'old'
    }
    age = letter_to_age.get(choice, "None")


    chat_hair = (
        "Select the hair color of the person in this video from the hair color categories listed.\n "
        "A: White\nB: Blonde\nC: Black\nD: Red"
        "Please only answer with the CAPTIAL letter of your choice. Again, captial letter from the 4 choices. "
    )
    # Run through model 
    response, chat_history = model.chat(tokenizer, '', chat_hair, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config={'do_sample':False})
    # Regex to extract 
    m = re.search(r"\b[A-C]\b", response)
    choice = m.group(0) if m else "None"
    letter_to_hair = {
        'A': 'white',
        'B': 'blonde',
        'C': 'black',
        'D': 'red'
    }
    hair = letter_to_hair.get(choice, "None")
    

    chat_body = (
        "Does the person in this video have a thin or a wide body? "
        "Please answer with only a 'thin' or a 'wide' reponse. "
    )
    # Run through model 
    response, chat_history = model.chat(tokenizer, '', chat_body, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config={'do_sample':False})
    # Regex to find female inside the response
    pattern_body = re.compile(r'\bthin\b', re.IGNORECASE) 
    body = 'thin' if bool(pattern_body.search(response)) else 'wide'



    return gender, race, age, hair, body

def process_dir(dir, out_file, model, tokenizer, num_action_inferences):
    """
    Processes all the scene clips. 
    Saves the results of inference of InternVideo2 to out_file for:
        - If a human is in the video.
        - if action is being performed or not.
        - The 5 bias types
    
    dir = your/path/to/Key-Folder
    Assumed Key-Folder Struct:
    Key-Folder
        |
        -- Action
                |
                Video-Title
                    -- Video-Scenes-000.mp4
                    -- Video-Scenes-001.mp4
                    ...
                    -- Video-Scenes-999.mp4
    
    Args:
    dir (string): Directory to be processed
    out_file (string): name of the csv file to write out to
    model: hugging face model
    tokenizer: hugging face tokenizer
    num_action_inferences (int): number of times to repeat the action inference, must be odd.
    """
    path_to_len = len(dir.split(os.path.sep))

    # Open csv to write
    with open(out_file, "w", newline="") as out:
        # Create writer and write headers
        writer = csv.writer(out, delimiter=',')
        headers = ["file", "action", "does_contain_person", "is_action_happening", "inference_len", "Gender", "Race", "Age", "Hair", "Body"]
        writer.writerow(headers)


        # Walk through each action folder action folder
        for i, folder in enumerate(os.walk(dir)):
            if i==0: # skip first
                continue

            # Top level path to action folder
            top_level_dir = os.path.join(os.getcwd(), folder[0])
            
            # Get length of path to determine if it is an action folder or a folder of video clips
            path_list = folder[0].split(os.path.sep)
            # Extract the action from folder name
            if(len(path_list) < path_to_len + 2):
                action = path_list[-1]
                print(f"action: {action} starting")
                continue
            
            # Obtain and write InternVideo2 output for each video file in action folder
            for j, file in enumerate(folder[2]):
                # Format path and start timer
                path = os.path.join(top_level_dir, file)
                # Start timer
                start_time = time.time()
                
                # Ensure the video contains a person
                contains_person, video_tensor = inference_contains_person(model, tokenizer, path)
                if not contains_person:
                    end_time = time.time()
                    inference_time = end_time - start_time
                    row = [os.path.join(folder[0], file), action, contains_person, False, inference_time, "None", "None", "None", "None", "None"]
                    writer.writerow(row) # Write results to csv
                    continue
                
                # Ensure the action is happening
                yes = 0
                no = 0
                for i in range(num_action_inferences):
                    ret = inference_action(model, tokenizer, video_tensor, action)
                    if ret:
                        yes += 1
                    else:
                        no += 1
                
                action_found = True if yes>no else False
                if not action_found:
                    end_time = time.time()
                    inference_time = end_time - start_time
                    row = [os.path.join(folder[0], file), action, contains_person, action_found, inference_time, "None", "None", "None", "None", "None"]
                    writer.writerow(row) # Write results to csv
                    continue

                # Get biases in video
                gender, race, age, hair, body = inference_bias(model, tokenizer, video_tensor)
                row = [os.path.join(folder[0], file), action, contains_person, action_found, inference_time, gender, race, age, hair, body]
                writer.writerow(row) # Write results to csv

            print(f"a {action} video directory processed.")

def main(args):
    print("Script start.")
    start_time = time.time()
    
    # Get Device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Get token from .env file
    load_dotenv()
    auth = os.getenv("HF")
    login(auth)

    assert args.num_shots % 2 == 1, f"num_shots is not odd"

    # Load tokenizer and model from hugging face
    tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        'OpenGVLab/InternVideo2-Chat-8B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    # Process the directory
    logging.set_verbosity_error()
    process_dir(args.dir, args.out, model, tokenizer, args.num_shots)
    
    # Get run time
    stop_time = time.time()
    seconds = int(stop_time - start_time)
    days, rem = divmod(seconds, 86400)       # 86400 sec = 1 day
    hours, rem = divmod(rem, 3600)           # 3600 sec = 1 hour
    minutes, secs = divmod(rem, 60)          # 60 sec = 1 minute
    formatted_time = f"{days}:{hours:02}:{minutes:02}:{secs:02}"
    
    print(f"Script complete. Time={formatted_time}")


if __name__ == "__main__":
    # Arguments 
    parser = argparse.ArgumentParser()
    
    # dir = your/path/to/folder --> folder(Actions(Videos(scene.mp4)))
    parser.add_argument("dir", help="The path of the directory of the videos you want to test")
    parser.add_argument("--out", help="Name of the file you want out to", default=f"actions-out-{time.time()}.csv")
    parser.add_argument("--num_shots", type=int, help="Number of times you want to run inference for the action category", default=1)
    
    args = parser.parse_args()
    main(args)


