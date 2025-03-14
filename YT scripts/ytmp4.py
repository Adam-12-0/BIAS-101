import os

def download_youtube_video(url, output_path='.'):
    """
    Downloads a YouTube video as an MP4 file.

    Parameters:
    - url: str, the YouTube video URL.
    - output_path: str, the directory where the video will be saved.
    """
    command = f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "{output_path}/%(title)s.%(ext)s" {url}'
    os.system(command)

# Example usage
download_youtube_video('https://youtu.be/52AOPbn1BAA?si=ElIBCKxhrDyEeCf8', r'C:\Users\AdamB\OneDrive\Documents\BIAS-101')
