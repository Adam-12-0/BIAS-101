import os
import pandas as pd
import matplotlib.pyplot as plt

HIGH_CONTRAST_COLORS = [
    "#7F7FFF",  # Light Blue
    "#FF7F7F",  # Light Red
    "#000000",  # Black
    "#7FFF7F",  # Light Green
    "#AA7FEE",  # Light Purple
    "#FFC87F",  # Light Orange
]

def extract_category_name(video_name: str) -> str:
    return video_name.split('_')[1]

def compute_dominance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    category_stats = {}

    for category, group in df.groupby('category'):
        subcategory_counts = group['final'].value_counts(normalize=True) * 100
        most_represented = subcategory_counts.idxmax()
        most_represented_score = subcategory_counts.max()

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
            df['category'] = df['video_name'].apply(extract_category_name)

            scores_df = compute_dominance_ratio(df)
            fig, ax = plt.subplots(figsize=(12, 6))

            unique_subcategories = sorted(scores_df.columns)
            colors = {sub: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, sub in enumerate(unique_subcategories)}

            bottom = pd.Series([0] * len(scores_df), index=scores_df.index)
            for subcategory in unique_subcategories:
                ax.bar(scores_df.index, scores_df[subcategory], bottom=bottom,
                       color=colors.get(subcategory, 'lime'), label=subcategory)
                bottom += scores_df[subcategory]

            balance_line = 100 / len(unique_subcategories)
            average_dominance = scores_df.max(axis=1).mean()

            ax.axhline(y=balance_line, color='black', linestyle='dashed', linewidth=2, label='Balance')
            ax.axhline(y=average_dominance, color='red', linestyle='solid', linewidth=2, label=f'Average: {average_dominance:.2f}')

            ax.set_xticklabels(scores_df.index, rotation=45, ha='right', fontsize=5)
            ax.set_title(f'{model_name.upper()} - Dominance Ratio Visualization')
            ax.set_xlabel('Action Categories', fontsize=12)
            ax.set_ylabel('Dominance Ratio (0 - 100)', fontsize=12)
            ax.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{output_folder}{file.split("_")[1].split(".")[0]}_dr.png')
            plt.close()


def main():
    process_model_data('clip')
    process_model_data('llava')


if __name__ == "__main__":
    main()
