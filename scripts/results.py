import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use("dark_background")  # Dark mode for plots


HIGH_CONTRAST_COLORS = [
    "#FF1A1A",  # Neon Red
    "#1A1AFF",  # Neon Blue
    "#1AFF1A",  # Neon Green
    "#A300A3",  # Neon Purple
    "#FF751A",  # Neon Orange
    "#FFFFFF",  # White    
]

def load_and_prepare_data(model_name: str, bias_type: str):
    """
    Loads and prepares data from the gender or race CSV file.
    Returns a DataFrame of categories with percentage representation.
    """
    input_file = f'data/{model_name}/{model_name}_{bias_type}.csv'

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} does not exist.")

    data = pd.read_csv(input_file)
    data['category'] = data['video_name'].apply(lambda x: x.split('_')[1])

    # Count occurrences of each subcategory within each action category
    category_counts = data.groupby(['category', 'final']).size().reset_index(name='count')
    category_counts['percentage'] = category_counts.groupby('category')['count'].transform(lambda x: (x / x.sum()))

    # Pivot the table
    category_df = category_counts.pivot(index='category', columns='final', values='percentage').fillna(0)

    return category_df


def plot_stacked_bar_chart(model_name: str, bias_type: str, category_df: pd.DataFrame):
    """
    Plots stacked bar charts for the given model (CLIP or LLaVA) and bias type (Gender or Race).
    """
    plt.figure(figsize=(16, 10))

    # Generate consistent colors for each subcategory
    unique_subcategories = category_df.columns
    colors = {sub: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, sub in enumerate(unique_subcategories)}

    # Plot each subcategory as a part of the stacked bar
    bottom = pd.Series([0] * len(category_df), index=category_df.index)
    for subcategory in unique_subcategories:
        plt.bar(category_df.index, category_df[subcategory], bottom=bottom, color=colors[subcategory], label=subcategory)
        bottom += category_df[subcategory]

    # Graph appearance
    plt.title(f'{model_name.upper()} - {bias_type.capitalize()} Bias Distribution', fontsize=16, color='white')
    plt.xlabel('Action Categories', fontsize=12, color='white')
    plt.ylabel('Percentage of Videos (0 to 1)', fontsize=12, color='white')
    plt.xticks(rotation=90, fontsize=5)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()

    # Save plot
    output_folder = f'visualizations/{model_name}/'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f'{output_folder}/{bias_type}_results.png')
    plt.close()


def main():
    for model_name in ['clip', 'llava']:
        for bias_type in ['gender', 'race']:
            category_df = load_and_prepare_data(model_name, bias_type)
            plot_stacked_bar_chart(model_name, bias_type, category_df)


if __name__ == "__main__":
    main()
