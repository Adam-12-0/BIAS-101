import os
import subprocess
import json

def download_youtube_playlist_as_mp4(playlist_url, base_output_path='.'):
    """
    Downloads all videos from a YouTube playlist as MP4 files into a specified folder,
    naming each file after the video title, and skipping already existing files.

    Parameters:
    - playlist_url: str, the URL of the YouTube playlist.
    - base_output_path: str, the base directory where the playlist folder will be created.
    """
    # Retrieve playlist metadata to get the playlist title
    result = subprocess.run(
        ['yt-dlp', '--flat-playlist', '--dump-single-json', playlist_url],
        capture_output=True, text=True
    )
    playlist_info = json.loads(result.stdout)
    playlist_title = playlist_info.get('title', 'Unnamed_Playlist')

    # Create the output directory named after the playlist title
    output_path = os.path.join(base_output_path, playlist_title)
    os.makedirs(output_path, exist_ok=True)

    # Path to the download archive file
    archive_path = os.path.join(output_path, 'downloaded_videos.txt')

    # yt-dlp command to download videos as MP4
    command = [
        'yt-dlp',
        '--ignore-errors',
        '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        '--output', f'{output_path}/%(title)s.%(ext)s',
        '--download-archive', archive_path,
        '--yes-playlist',
        playlist_url
    ]
    subprocess.run(command)

# Example usage
playlist_url = 'https://youtube.com/playlist?list=PLfOIoB55XlgdDXWQoodOO_d3VZfHD3MVz'
base_output_directory = r'C:\Users\AdamB\OneDrive\Documents\BIAS-101'
download_youtube_playlist_as_mp4(playlist_url, base_output_directory)
