from scenedetect import detect, AdaptiveDetector, video_splitter

def find_scenes(input_path, print_times=False): 
    """
    Finds the scenes in an mp4 video.

    Args:
        input_path (string): file path to the mp4 file you want the scenes to.
        print_times (boolean): a boolean representing wheather you want the time stamps printed.

    Returns:
        scene_list: a list of tuples cantaining a start FrameTimecode and end FrameTimecode
    """
    
    scene_list = detect(input_path, AdaptiveDetector())
    if(print_times):
        print_scene_changes(scene_list)
    return scene_list

def clip_scenes(input_path, scene_list, output_path):
    """
    Split the input video based on the scene list passed in

    Args:
        input_path (string): file path to the mp4 file you want the split into clips.
        scene_list (list): a list of tuples cantaining a start FrameTimecode and end FrameTimecode.
        output_path (string): file path the the mp4 file you want the clips saved to.
    """
    video_splitter.split_video_ffmpeg(input_path, scene_list, output_path)

def print_scene_changes(scene_list):
    """
    Prints the scene list in a human readable format.

    Args:
        scene_list (list): a list of tuples cantaining a start FrameTimecode and end FrameTimecode.
    """
    
    for i, (start_time, end_time) in enumerate(scene_list, start=1):
        print(f"Scene {i}: Start {start_time.get_timecode()} [Frame {start_time.get_frames()}], "
              f"End {end_time.get_timecode()} [Frame {end_time.get_frames()}]")


# Example usage
video_path = "Survivor.mp4"
scene_list = find_scenes(video_path, True)
clip_scenes(video_path, scene_list, "clips/")

