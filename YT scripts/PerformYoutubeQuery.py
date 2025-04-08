import os
import pandas as pd
import requests
import time
import argparse
from googleapiclient.discovery import build
from datetime import datetime
from googleapiclient.errors import HttpError

# Configuration
QUERIES_FILE = "search_queries.csv"
RESULTS_DIR = "youtube_results"
MAX_RESULTS_PER_QUERY = 50 # 50 is max allowed
DELAY_BETWEEN_QUERIES = .5  # seconds, to avoid hitting API rate limits
# Default API keys list - replace with your actual API keys
API_KEYS = [
    'AIzaSyCvqFjB9_6lyYkSWWgTthiz9nOCTlZzMc4',
    # Add more API keys here
]


def setup_results_directory(base_dir=RESULTS_DIR):
    """
    Create directory structure for saving results.

    Args:
        base_dir (str): Base directory for results

    Returns:
        str: Path to the results directory
    """
    # Create directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created results directory: {base_dir}")

    return base_dir


class YouTubeAPIManager:
    """
    Manages YouTube API keys and handles rate limit errors.
    """

    def __init__(self, api_keys):
        """
        Initialize with a list of API keys.

        Args:
            api_keys (list): List of YouTube API keys
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.youtube = self._build_youtube_client()

    def _build_youtube_client(self):
        """
        Build a YouTube API client with the current API key.

        Returns:
            object: YouTube API client
        """
        return build('youtube', 'v3', developerKey=self.get_current_key())

    def get_current_key(self):
        """
        Get the current API key.

        Returns:
            str: Current API key
        """
        return self.api_keys[self.current_key_index]

    def rotate_key(self):
        """
        Rotate to the next API key.

        Returns:
            bool: True if successfully rotated, False if no more keys
        """
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.youtube = self._build_youtube_client()
            print(f"Switched to API key {self.current_key_index + 1}/{len(self.api_keys)}")
            return True
        else:
            print("WARNING: No more API keys available!")
            return False

    def is_rate_limit_error(self, error):
        """
        Check if an error is due to API rate limiting.

        Args:
            error: Error object from API call

        Returns:
            bool: True if rate limit error, False otherwise
        """
        if isinstance(error, HttpError):
            if error.resp.status in [403, 429]:
                error_details = error._get_reason()
                rate_limit_messages = ["quotaExceeded", "userRateLimitExceeded", "dailyLimitExceeded",
                                       "rateLimitExceeded"]
                return any(msg in error_details for msg in rate_limit_messages)
        return False

    def execute_with_retry(self, request):
        """
        Execute an API request with key rotation on rate limit errors.

        Args:
            request: YouTube API request object

        Returns:
            dict: API response

        Raises:
            Exception: If all API keys are exhausted or other errors occur
        """
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            try:
                return request.execute()
            except Exception as e:
                attempts += 1
                if self.is_rate_limit_error(e):
                    print(f"Rate limit reached for API key {self.current_key_index + 1}")
                    if not self.rotate_key():
                        raise Exception("All API keys have reached their rate limits") from e
                else:
                    # Not a rate limit error, re-raise
                    raise e

        raise Exception("Failed after trying all API keys")


def search_youtube_videos(api_manager, query, max_results=10):
    """
    Search YouTube for videos matching the query.

    Args:
        api_manager (YouTubeAPIManager): API manager instance
        query (str): Search term
        max_results (int): Maximum number of results to return

    Returns:
        list: List of video data dictionaries
    """
    request = api_manager.youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    )

    response = api_manager.execute_with_retry(request)

    # Get the video IDs
    video_ids = [item['id']['videoId'] for item in response['items']]

    # Get additional metadata for these videos
    if video_ids:
        videos_request = api_manager.youtube.videos().list(
            part='snippet,contentDetails,statistics,topicDetails',
            id=','.join(video_ids)
        )
        videos_response = api_manager.execute_with_retry(videos_request)

        # Create a mapping of video IDs to their additional metadata
        video_metadata = {item['id']: item for item in videos_response['items']}

        # Add additional metadata to the original search results
        for item in response['items']:
            video_id = item['id']['videoId']
            if video_id in video_metadata:
                item['metadata'] = video_metadata[video_id]

    return response['items']


def download_thumbnail(url, filename):
    """
    Download an image from a URL and save it to a file.

    Args:
        url (str): URL of the image
        filename (str): Path to save the image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def parse_duration(duration_str):
    """
    Parse ISO 8601 duration format to seconds.

    Args:
        duration_str (str): Duration string in ISO 8601 format (e.g., 'PT1H2M3S')

    Returns:
        int: Duration in seconds
    """
    duration = 0
    # Remove 'PT' prefix
    time_str = duration_str.replace('PT', '')

    # Extract hours
    if 'H' in time_str:
        hours, time_str = time_str.split('H')
        duration += int(hours) * 3600

    # Extract minutes
    if 'M' in time_str:
        minutes, time_str = time_str.split('M')
        duration += int(minutes) * 60

    # Extract seconds
    if 'S' in time_str:
        seconds = time_str.replace('S', '')
        duration += int(seconds)

    return duration


def process_video_data(videos, body, age, gender, action, action_dir):
    """
    Process video data and download thumbnails.

    Args:
        videos (list): List of video data from YouTube API
        body (str): Body type used in the search query
        age (str): Age group used in the search query
        gender (str): Gender used in the search query
        action (str): Action used in the search query
        action_dir (str): Directory for the action category

    Returns:
        list: List of processed video data dictionaries
    """
    processed_data = []

    # Create a single thumbnails directory under the action directory
    thumbnails_dir = os.path.join(action_dir, 'thumbnails')
    if not os.path.exists(thumbnails_dir):
        os.makedirs(thumbnails_dir)

    for video in videos:
        video_id = video['id']['videoId']
        title = video['snippet']['title']
        link = f"https://www.youtube.com/watch?v={video_id}"

        # Get basic snippet data
        description = video['snippet'].get('description', '')

        # Dictionary to store our processed data
        video_data = {
            'Body': body,
            'Age': age,
            'Gender': gender,
            'Action': action,
            'Search Query': f"{body} {age} {gender} {action}",
            'Title': title,
            'Video ID': video_id,
            'Link': link,
            'Description': description
        }

        # Additional metadata if available
        if 'metadata' in video:
            metadata = video['metadata']

            # Content details
            if 'contentDetails' in metadata:
                # Parse duration from ISO 8601 format
                duration_str = metadata['contentDetails'].get('duration', '')
                duration_seconds = parse_duration(duration_str) if duration_str else 0

                video_data.update({
                    'Duration (seconds)': duration_seconds
                })

            # Statistics
            if 'statistics' in metadata:
                video_data.update({
                    'View Count': int(metadata['statistics'].get('viewCount', 0)),
                    'Like Count': int(metadata['statistics'].get('likeCount', 0)),
                    'Comment Count': int(metadata['statistics'].get('commentCount', 0))
                })

            # Topic details
            if 'topicDetails' in metadata and 'topicCategories' in metadata['topicDetails']:
                topics = metadata['topicDetails']['topicCategories']
                # Extract the category name from the URL (e.g., https://en.wikipedia.org/wiki/Music -> Music)
                topic_names = [t.split('/')[-1].replace('_', ' ') for t in topics]
                video_data['Topics'] = ', '.join(topic_names)

        # Get highest quality thumbnail available
        thumbnails = video['snippet']['thumbnails']
        if 'high' in thumbnails:
            thumbnail_url = thumbnails['high']['url']
        elif 'medium' in thumbnails:
            thumbnail_url = thumbnails['medium']['url']
        else:
            thumbnail_url = thumbnails['default']['url']

        # Create a unique filename for the thumbnail including demographic info to avoid conflicts
        # Format: videoID_body_age_gender.jpg
        img_filename = f"{video_id}_{body}_{age}_{gender}.jpg".replace(" ", "_")
        img_path = os.path.join(thumbnails_dir, img_filename)

        # Create a relative path for storage in CSV (easier to move directories later)
        relative_img_path = os.path.join('thumbnails', img_filename)

        # Download the thumbnail
        success = download_thumbnail(thumbnail_url, img_path)

        # Add thumbnail path to video data
        video_data['Thumbnail Path'] = relative_img_path if success else 'Download failed'

        processed_data.append(video_data)

    return processed_data


def save_results_incrementally(results, action, results_dir, mode='a'):
    """
    Save results incrementally to CSV file.

    Args:
        results (list): List of results to save
        action (str): Action category
        results_dir (str): Base directory for results
        mode (str): File open mode ('a' for append, 'w' for write)

    Returns:
        str: Path to the saved CSV file
    """
    if not results:
        return None

    # Create action directory if it doesn't exist
    action_safe_name = action.replace(' ', '_')
    action_dir = os.path.join(results_dir, action_safe_name)

    if not os.path.exists(action_dir):
        os.makedirs(action_dir)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Define CSV path
    csv_path = os.path.join(action_dir, f"{action_safe_name}_results.csv")

    # Check if file exists to determine if we need headers
    file_exists = os.path.isfile(csv_path)

    # Save to CSV
    if mode == 'w' or not file_exists:
        df.to_csv(csv_path, index=False)
    else:  # Append without headers if file exists
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    return csv_path


def load_search_queries(file_path=QUERIES_FILE, subset=None):
    """
    Load search queries from CSV file.

    Args:
        file_path (str): Path to the CSV file with queries
        subset (int): Optional number of queries to process (for testing)

    Returns:
        pd.DataFrame: DataFrame containing the queries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Queries file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} queries from {file_path}")

    # If subset is specified, take only that many queries
    if subset and subset > 0:
        df = df.head(subset)
        print(f"Processing subset of {subset} queries")

    return df


def get_completed_queries(results_dir):
    """
    Get a list of already completed queries from existing CSV files.

    Args:
        results_dir (str): Directory where results are stored

    Returns:
        set: Set of already processed "body age gender action" combinations
    """
    completed = set()

    # Check if the directory exists
    if not os.path.exists(results_dir):
        return completed

    # Check each action directory for result CSVs
    for action_dir in os.listdir(results_dir):
        action_path = os.path.join(results_dir, action_dir)

        # Skip if not a directory
        if not os.path.isdir(action_path):
            continue

        # Check for results CSV
        csv_path = os.path.join(action_path, f"{action_dir}_results.csv")
        if os.path.exists(csv_path):
            try:
                # Load the CSV and get the unique search queries
                df = pd.read_csv(csv_path)
                if 'Search Query' in df.columns:
                    completed.update(df['Search Query'].unique())
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    return completed


def process_queries(queries_df, results_dir, api_manager, max_results=MAX_RESULTS_PER_QUERY,
                    delay=DELAY_BETWEEN_QUERIES, resume=True):
    """
    Process each query in the DataFrame and save results incrementally.

    Args:
        queries_df (pd.DataFrame): DataFrame with search queries
        results_dir (str): Directory to save results
        api_manager (YouTubeAPIManager): API manager instance
        max_results (int): Maximum results per query
        delay (int): Delay between queries in seconds
        resume (bool): Whether to skip already processed queries
    """
    total_queries = len(queries_df)
    processed = 0
    errors = 0

    # Get already completed queries if resuming
    completed_queries = set()
    if resume:
        completed_queries = get_completed_queries(results_dir)
        if completed_queries:
            print(f"Found {len(completed_queries)} already processed queries")

    print(f"Starting batch processing of {total_queries} queries...")
    print(f"Using {len(api_manager.api_keys)} API keys")

    # Process each query
    for index, row in queries_df.iterrows():
        query = row['full_query']
        body = row['body']
        age = row['age']
        gender = row['gender']
        action = row['action']

        # Check if this query has already been processed
        search_query = f"{body} {age} {gender} {action}"
        if resume and search_query in completed_queries:
            print(f"[{index + 1}/{total_queries}] Skipping (already processed): '{query}'")
            processed += 1
            continue

        # Create directory for the action category
        action_safe_name = action.replace(' ', '_')
        action_dir = os.path.join(results_dir, action_safe_name)
        if not os.path.exists(action_dir):
            os.makedirs(action_dir)

        try:
            print(
                f"[{index + 1}/{total_queries}] Searching: '{query}' (using API key {api_manager.current_key_index + 1})")

            # Call the YouTube search function
            videos = search_youtube_videos(api_manager, query, max_results=max_results)

            if videos:
                # Process the videos and get the data
                results = process_video_data(videos, body, age, gender, action, action_dir)

                # Save results incrementally
                csv_path = save_results_incrementally(results, action, results_dir)

                print(f"  ✓ Found {len(videos)} videos")
                print(f"  ✓ Saved results to {csv_path}")
            else:
                print(f"  ✗ No videos found")

            processed += 1

            # Delay between queries to avoid rate limiting
            if index < total_queries - 1:  # If not the last query
                print(f"  Waiting {delay} seconds before next query...")
                time.sleep(delay)

        except Exception as e:
            if isinstance(e, HttpError) and api_manager.is_rate_limit_error(e):
                print(f"  ✗ Rate limit reached for current API key")
                if not api_manager.rotate_key():
                    print("  ✗ All API keys have reached their rate limits. Stopping.")
                    break

                # Retry the same query with the new key after a short delay
                print(f"  Retrying query with new API key...")
                time.sleep(2)  # Short delay before retrying
                index -= 1  # Adjust index to retry the same query
                continue
            else:
                print(f"  ✗ Error processing query '{query}': {e}")
                errors += 1

            # Still wait before the next query even if there was an error
            if index < total_queries - 1:
                print(f"  Waiting {delay} seconds before next query...")
                time.sleep(delay)

    print(f"\nBatch processing complete!")
    print(f"Processed: {processed}/{total_queries} queries")
    print(f"Errors: {errors}")
    print(f"Results saved to: {results_dir}")


def main():
    """Main function to run the batch search script."""
    parser = argparse.ArgumentParser(description="Process YouTube searches for UCF-101 debiasing")
    parser.add_argument("--queries", default=QUERIES_FILE, help="CSV file with search queries")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS_PER_QUERY, help="Max results per query")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_QUERIES, help="Delay between queries (seconds)")
    parser.add_argument("--subset", type=int, help="Process only a subset of queries (for testing)")
    parser.add_argument("--api-keys", nargs='+', help="List of YouTube API keys")
    parser.add_argument("--api-keys-file", help="File containing YouTube API keys (one per line)")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already processed queries")

    args = parser.parse_args()

    # Get API keys from command line or file
    api_keys = API_KEYS  # Default

    if args.api_keys:
        api_keys = args.api_keys
        print(f"Using {len(api_keys)} API keys from command line")
    elif args.api_keys_file:
        try:
            with open(args.api_keys_file, 'r') as f:
                api_keys = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(api_keys)} API keys from {args.api_keys_file}")
        except Exception as e:
            print(f"Error loading API keys from file: {e}")
            print("Falling back to default API keys")

    if not api_keys:
        print("No API keys provided! Please provide at least one API key.")
        return

    # Initialize API manager
    api_manager = YouTubeAPIManager(api_keys)

    # Setup results directory
    results_dir = setup_results_directory(args.results_dir)

    # Load queries
    queries_df = load_search_queries(args.queries, args.subset)

    # Process queries
    process_queries(
        queries_df,
        results_dir,
        api_manager,
        args.max_results,
        args.delay,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()