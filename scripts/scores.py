import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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

def compute_dominance_ratio_and_chi_square(df: pd.DataFrame) -> pd.DataFrame:
    category_stats = {}

    for category, group in df.groupby('category'):
        subcategory_counts = group['final'].value_counts()
        most_represented = subcategory_counts.idxmax()
        most_represented_score = subcategory_counts.max()
        total_count = subcategory_counts.sum()

        N = len(subcategory_counts)
        dominance_ratio = most_represented_score / (total_count * N) if total_count > 0 else 0

        expected_counts = [total_count / N] * N
        observed_counts = subcategory_counts.values

        if len(observed_counts) == N:
            epsilon = 1e-10
            adjusted_expected_counts = [max(count, epsilon) for count in expected_counts]
            chi_square = sum(((observed_counts - adjusted_expected_counts) ** 2) / adjusted_expected_counts)
        else:
            chi_square = 0

        category_stats[category] = {
            'Dominant Class': most_represented,
            'Dominance Ratio': dominance_ratio * 100,
            'Chi-Square': chi_square
        }

    return pd.DataFrame.from_dict(category_stats, orient='index').fillna(0)

def plot_and_save(data, model_name, metric_name, bias_type, output_folder):
    plt.figure(figsize=(12, 6))

    handles = []

    if metric_name == f'{bias_type.capitalize()} DR':
        unique_categories = sorted(data[f'{bias_type.capitalize()} Class'].unique())
        colors = {sub: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, sub in enumerate(unique_categories)}
        bar_colors = [colors[class_name] if class_name in colors else 'gray' for class_name in data[f'{bias_type.capitalize()} Class']]
        bars = plt.bar(data.index, data[metric_name], color=bar_colors)

        # Add legend handles for each class
        for key in colors.keys():
            handles.append(plt.Line2D([0], [0], color=colors[key], lw=4, label=key))
    else:
        bars = plt.bar(data.index, data[metric_name], color='blue')

    # Add average line
    avg = data[metric_name].mean()
    avg_line = plt.axhline(y=avg, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg:.2f}')
    handles.append(avg_line)

    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.title(f'{model_name.upper()} - {metric_name} Visualization')
    plt.xlabel('Action Categories', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.tight_layout()

    # Show full legend
    plt.legend(handles=handles, loc='upper right', fontsize=10)

    plot_name = f'{bias_type}_dr.png' if 'DR' in metric_name else f'{bias_type}_chi.png'
    plt.savefig(f'{output_folder}/{plot_name}')
    plt.close()

def process_model_data(model_name: str):
    input_folder = f'data/{model_name}/'
    output_folder_root = f'visualizations/{model_name}/'

    combined_results = pd.DataFrame()

    for file in os.listdir(input_folder):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)[['video_name', 'final']]
        df['category'] = df['video_name'].apply(extract_category_name)

        results_df = compute_dominance_ratio_and_chi_square(df)

        # Correct bias_type extraction: get last part before .csv
        bias_type = file.replace('.csv', '').split('_')[-1]
        results_df.columns = [f'{bias_type.capitalize()} Class', f'{bias_type.capitalize()} DR', f'{bias_type.capitalize()} Chi-Square']

        combined_results = pd.concat([combined_results, results_df], axis=1)

        output_folder = os.path.join(output_folder_root, bias_type)
        os.makedirs(output_folder, exist_ok=True)

        plot_and_save(results_df, model_name, f'{bias_type.capitalize()} DR', bias_type, output_folder)
        plot_and_save(results_df, model_name, f'{bias_type.capitalize()} Chi-Square', bias_type, output_folder)

    combined_results.index.name = 'Category'
    output_csv = f'data/{model_name}_scores.csv'
    combined_results.to_csv(output_csv)
    print(f"Scores saved to: {output_csv}")

def main():
    model_folders = [f for f in os.listdir('data') if os.path.isdir(os.path.join('data', f))]
    for model in model_folders:
        process_model_data(model)

if __name__ == "__main__":
    main()
