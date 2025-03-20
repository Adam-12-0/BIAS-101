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


def extract_category_name(video_name: str) -> str:
    return video_name.split('_')[1]


def compute_dominance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    category_stats = {}

    for category, group in df.groupby('category'):
        total_samples = len(group)
        if total_samples == 0:
            continue

        subcategory_counts = group['final'].value_counts(normalize=True) * 100
        most_represented = subcategory_counts.idxmax()
        most_represented_score = subcategory_counts.max()

        # Prepare dominance distribution where only the dominant category is visible
        dominance_distribution = {subcategory: 0 for subcategory in subcategory_counts.index}
        dominance_distribution[most_represented] = most_represented_score

        category_stats[category] = dominance_distribution

    return pd.DataFrame.from_dict(category_stats, orient='index').fillna(0)


def process_model_data(model_name: str):
    input_folder = f'data/{model_name}/'
    output_folder = f'visualizations/{model_name}/'
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))[['video_name', 'final']]
            if 'video_name' in df.columns:
                df['category'] = df['video_name'].apply(extract_category_name)

            scores_df = compute_dominance_ratio(df)

            plt.figure(figsize=(12, 6))

            # Generate consistent colors for each subcategory
            unique_subcategories = sorted(scores_df.columns)  # Sorting to maintain consistent legend order
            colors = {sub: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, sub in enumerate(unique_subcategories)}

            bottom = pd.Series([0] * len(scores_df), index=scores_df.index)

            # Plot each subcategory as a stacked bar, only the dominant class will be visible
            for subcategory in unique_subcategories:
                plt.bar(scores_df.index, scores_df[subcategory], bottom=bottom,
                        color=colors.get(subcategory, 'lime'), label=subcategory)
                bottom += scores_df[subcategory]

            plt.xticks(rotation=90, fontsize=5)
            plt.title(f'{model_name.upper()} - Dominance Ratio Visualization', fontsize=14, color='white')
            plt.xlabel('Action Categories', fontsize=12, color='white')
            plt.ylabel('Dominance Ratio (0 - 100)', fontsize=12, color='white')

            # Add consistent legend
            handles = [plt.Line2D([0], [0], color=colors[sub], lw=4) for sub in unique_subcategories]
            plt.legend(handles, unique_subcategories, loc='upper right', fontsize=8)

            plt.tight_layout()

            plot_type = file.split('_')[1].split('.')[0]
            plt.savefig(f'{output_folder}{plot_type}_dr.png')
            plt.close()


def main():
    process_model_data('clip')
    process_model_data('llava')


if __name__ == "__main__":
    main()
