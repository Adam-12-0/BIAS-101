import torch
from transformers import AutoProcessor, AutoModel
import cv2
import numpy as np

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

def inference(model, processor, video, action_desc, idx, print_scores):
    """ 
    Performs inference with the xclip model and returns if the action is being performed or not.
    
    Args:
        Model (Hugging Face model): The XCLIP model.
        processor (Hugging Face processor): The XCLIP Processor.
        action_desc (list) : A list of strings of action descriptions.
        idx (int) : The index in action_desc of the description representing the action in question.
        print_scores (bool) : Whether or not to print the probabilities.
    
    Return: Boolean determining whether the model thinks the action is happening or not. 
    """
    
    # Convert video to list of numpy arrays of each frame
    video_list = vid_to_np(video)
    video_list = subsample_frames(video_list) # Subsample or trim the frames to match the expected number (32)

    # Preprocess the video frames and text
    inputs = processor(text=action_desc, videos=video_list, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_video = outputs.logits_per_video  # Similarity score between video and text
        probs = logits_per_video.softmax(dim=1)  # Convert logits to probabilities

    # Similarity score for action in question
    similarity_score = probs[0][idx].item()
    # Most likely text description
    max_val = max(probs[0])
    
    # Print scores
    if print_scores:
        print(probs)
        print(f"Similarity Score: {similarity_score:.4f}")
        print(max_val)


    # Return if the action is present
    if similarity_score == max_val:
        return True
    else:
        return False

def main():
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-ucf-16-shot")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch16-ucf-16-shot")
    video = './clips/Survivor-Scene-008.mp4' # a 6 sec clip of an Austrilan beauty pagent from austrailan survivor
    #desc = ["An Austrilan beauty pagent.", "NOT an Austrilan beauty pagent"] # Correct
    #desc = ["A baseball game", "NOT a baseball game"] # Corret
    desc = ["A concert", "NOT a concert"] # ncorrect
    print(inference(model, processor, video, desc, 0, True))

if __name__ == "__main__":
    main()