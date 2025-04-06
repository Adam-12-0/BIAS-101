from scenedetect import detect, AdaptiveDetector, video_splitter
import argparse
import os

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

def main(args):
    """
    Splits the passed dataset up based on detected scenes

    Args:
        args: flags from argparser, see --help for more info.
    """

    dataset = args.data
    # Example usage
    if(dataset == "sample"):
        video_path = "Survivor.mp4"
        scene_list = find_scenes(video_path, True)
        clip_scenes(video_path, scene_list, "clips/")
    # Test usage on original UCF101
    elif(dataset == "ucf"):
        in_path = "Full101"
        out_path = "Split101"
        for dir in os.walk(in_path):
            print("Script start.\n")
            for file in dir[2]:
                # Format the path stored and path to print to
                in_path_vid = os.path.join(in_path, file)
                out_path_vid = os.path.join(out_path, file[:-4])
                
                # Find scene markers and split videos
                scene_list = find_scenes(in_path_vid)
                clip_scenes(in_path_vid, scene_list, out_path_vid)
                
                # For comparsion see how many was in UCF101
                if(args.ucfpath != None):
                    ucf101_path = os.path.join(args.ucfpath, file[:-4])
                    orig_file_count = len([f for f in os.listdir(ucf101_path) if os.path.isfile(os.path.join(ucf101_path, f))])
                    sim = ((len(scene_list)/orig_file_count) * 100)
                    if len(scene_list) >= orig_file_count:
                        sim = 100 - sim
                    print(f"{in_path_vid} has beed completed.\nIt has {len(scene_list)} videos, UCF101 had {orig_file_count}. Similarity: {sim:.1f}%.\n")
                else:
                    print(f"{in_path_vid} has beed completed. It has {len(scene_list)} videos.")

        print("\nScript complete\n")
    # Passed dataset is not implemented yet.
    else:
        print("Dataset either not given or not supported.")

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="What dataset you want split options: sample, ucf", type=str)
    parser.add_argument("--ucfpath", help="Path to ucf 101 dataset on your computer", nargs='?', const='UCF-101', type=str)
    args = parser.parse_args()
    main(args)