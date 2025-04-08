import os
import pandas as pd
import csv
from datetime import datetime

# Configuration
OUTPUT_FILE = "search_queries.csv"

# Define the demographic parameters for debiasing
GENDERS = [
    "Men",
    "Women",
]

BODY = [
    "Skinny",
    "Fat",
]

AGE = [
    "Kid",
    "Young",
    "Old"
]

def get_action_categories(ucf_directory):
    """
    Extract action categories from the UCF-101 directory structure.

    Args:
        ucf_directory (str): Path to the UCF-101 directory

    Returns:
        list: List of action category names
    """
    try:
        # Get all subdirectories in the UCF directory
        action_categories = [d for d in os.listdir(ucf_directory)
                             if os.path.isdir(os.path.join(ucf_directory, d))]

        # Clean up category names - remove underscores and normalize
        action_categories = [cat.replace('_', ' ') for cat in action_categories]

        print(f"Found {len(action_categories)} action categories in UCF-101 dataset.")
        return sorted(action_categories)
    except Exception as e:
        print(f"Error reading UCF-101 directory: {e}")
        return []


def generate_search_queries(actions, ages, genders, bodies):
    """
    Generate search queries by combining age, gender, body type, and action.

    Args:
        actions (list): List of action categories
        ages (list): List of age groups to include
        genders (list): List of genders to include
        bodies (list): List of body types to include

    Returns:
        list: List of dictionaries containing the query components and full query
    """
    queries = []

    for action in actions:
        for age in ages:
            for gender in genders:
                for body in bodies:
                    # Format the query
                    query = f"{body} {age} {gender} {action}"

                    # Store the components for later analysis
                    queries.append({
                        'action': action,
                        'age': age,
                        'gender': gender,
                        'body': body,
                        'full_query': query
                    })

    print(f"Generated {len(queries)} unique search queries.")
    return queries


def save_queries_to_csv(queries, output_file=OUTPUT_FILE):
    """
    Save the generated queries to a CSV file.

    Args:
        queries (list): List of query dictionaries
        output_file (str): Path to save the CSV file
    """
    # Convert to DataFrame for easy CSV export
    df = pd.DataFrame(queries)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved queries to {output_file}")


def main():
    """Main function to run the query generation script."""
    # Configuration
    UCF_DIRECTORY = "UCF-101"  # Replace with your UCF-101 directory path

    # Get action categories from the UCF-101 directory
    actions = get_action_categories(UCF_DIRECTORY)

    if not actions:
        print("No action categories found. Exiting.")
        return

    # Preview some action categories
    print(f"Sample action categories: {', '.join(actions[:5])}...")

    # Generate the search queries using the predefined age, gender, and body lists
    queries = generate_search_queries(actions, AGE, GENDERS, BODY)

    print(f"Total queries: {len(queries)}")

    save_queries_to_csv(queries)


if __name__ == "__main__":
    main()