import os
import pandas as pd
import requests
import time
import argparse
from googleapiclient.discovery import build

# Configuration
QUERIES_FILE = "search_queries.csv"
RESULTS_DIR = "youtube_results"
MAX_RESULTS_PER_QUERY = 5
DELAY_BETWEEN_QUERIES = 10  # seconds, to avoid hitting API rate limits

API_KEY = 'AIzaSyCvqFjB9_6lyYkSWWgTthiz9nOCTlZzMc4'  # Replace with your actual API key


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


def search_youtube_videos(query, max_results=10):
    """
    Search YouTube for videos matching the query.

    Args:
        query (str): Search term
        max_results (int): Maximum number of results to return

    Returns:
        list: List of video data dictionaries
    """
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    )

    response = request.execute()
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


def process_video_data(videos, race, gender, action, thumbnails_dir):
    """
    Process video data and download thumbnails.

    Args:
        videos (list): List of video data from YouTube API
        race (str): Race used in the search query
        gender (str): Gender used in the search query
        action (str): Action used in the search query
        thumbnails_dir (str): Directory to save thumbnails

    Returns:
        list: List of processed video data dictionaries
    """
    processed_data = []

    for video in videos:
        video_id = video['id']['videoId']
        title = video['snippet']['title']
        link = f"https://www.youtube.com/watch?v={video_id}"

        # Get highest quality thumbnail available
        thumbnails = video['snippet']['thumbnails']
        if 'high' in thumbnails:
            thumbnail_url = thumbnails['high']['url']
        elif 'medium' in thumbnails:
            thumbnail_url = thumbnails['medium']['url']
        else:
            thumbnail_url = thumbnails['default']['url']

        # Create a unique filename for the thumbnail
        img_filename = f"{video_id}.jpg"
        img_path = os.path.join(thumbnails_dir, img_filename)

        # Download the thumbnail
        success = download_thumbnail(thumbnail_url, img_path)

        # Add query information and video data
        processed_data.append({
            'Race': race,
            'Gender': gender,
            'Action': action,
            'Search Query': f"{race} {gender} {action}",
            'Title': title,
            'Video ID': video_id,
            'Link': link,
            'Thumbnail Path': img_path if success else 'Download failed'
        })

    return processed_data


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


def process_queries(queries_df, results_dir, max_results=MAX_RESULTS_PER_QUERY, delay=DELAY_BETWEEN_QUERIES):
    """
    Process each query in the DataFrame and save results.

    Args:
        queries_df (pd.DataFrame): DataFrame with search queries
        results_dir (str): Directory to save results
        max_results (int): Maximum results per query
        delay (int): Delay between queries in seconds
    """
    total_queries = len(queries_df)
    processed = 0
    errors = 0

    # Dictionary to store results by action category
    action_results = {}

    print(f"Starting batch processing of {total_queries} queries...")

    # Process each query
    for index, row in queries_df.iterrows():
        query = row['full_query']
        race = row['race']
        gender = row['gender']
        action = row['action']

        # Create subdirectories for the action category
        action_dir = os.path.join(results_dir, action.replace(' ', '_'))
        thumbnails_dir = os.path.join(action_dir, 'thumbnails')

        if not os.path.exists(thumbnails_dir):
            os.makedirs(thumbnails_dir)

        try:
            print(f"[{processed + 1}/{total_queries}] Searching: '{query}'")

            # Call the YouTube search function
            videos = search_youtube_videos(query, max_results=max_results)

            if videos:
                # Process the videos and store the data
                results = process_video_data(videos, race, gender, action, thumbnails_dir)

                # Add to action-specific results (create list if first query for this action)
                if action not in action_results:
                    action_results[action] = []

                action_results[action].extend(results)

                print(f"  ✓ Found {len(videos)} videos")
            else:
                print(f"  ✗ No videos found")

            processed += 1

            # Delay between queries to avoid rate limiting
            if processed < total_queries:
                print(f"  Waiting {delay} seconds before next query...")
                time.sleep(delay)

        except Exception as e:
            print(f"  ✗ Error processing query '{query}': {e}")
            errors += 1

            # Continue with the next query
            continue

    # Save consolidated results by action category
    for action, results in action_results.items():
        if results:
            # Create DataFrame from all results for this action
            df = pd.DataFrame(results)

            # Save to a single CSV file for this action
            csv_path = os.path.join(results_dir, action.replace(' ', '_'), f"{action.replace(' ', '_')}_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(results)} results for '{action}' to {csv_path}")

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
    parser.add_argument("--delay", type=int, default=DELAY_BETWEEN_QUERIES, help="Delay between queries (seconds)")
    parser.add_argument("--subset", type=int, help="Process only a subset of queries (for testing)")
    parser.add_argument("--api-key", help="YouTube API key")

    args = parser.parse_args()

    # Override API key if provided
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
        print(f"Using provided API key")

    # Setup results directory
    results_dir = setup_results_directory(args.results_dir)

    # Load queries
    queries_df = load_search_queries(args.queries, args.subset)

    # Process queries
    process_queries(queries_df, results_dir, args.max_results, args.delay)


if __name__ == "__main__":
    main()
