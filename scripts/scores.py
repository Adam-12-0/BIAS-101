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

        # Calculate Dominance Ratio
        N = len(subcategory_counts)
        dominance_ratio = most_represented_score / (total_count * N) if total_count > 0 else 0

        # Calculate Chi-Square Score
        expected_counts = [total_count / N] * N
        observed_counts = subcategory_counts.values
        
        if len(observed_counts) == N:
            epsilon = 1e-10
            adjusted_expected_counts = [max(count, epsilon) for count in expected_counts]
            chi_square = sum(((observed_counts - adjusted_expected_counts) ** 2) / adjusted_expected_counts)
        else:
            chi_square = 0

        # Store results
        category_stats[category] = {
            'Dominant Class': most_represented,
            'Dominance Ratio': dominance_ratio,
            'Chi-Square': chi_square
        }

    return pd.DataFrame.from_dict(category_stats, orient='index').fillna(0)

def plot_and_save(data, model_name, metric_name, bias_type, output_folder):
    plt.figure(figsize=(12, 6))

    if metric_name == f'{bias_type.capitalize()} DR':
        colors = {sub: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, sub in enumerate(data[f'{bias_type.capitalize()} Class'].unique())}
        
        bar_colors = [colors[class_name] for class_name in data[f'{bias_type.capitalize()} Class']]
        plt.bar(data.index, data[metric_name], color=bar_colors)

        # Add custom legend for bar colors
        legend_handles = [plt.Line2D([0], [0], color=colors[key], lw=4, label=key) for key in colors.keys()]
        plt.legend(handles=legend_handles, loc='upper right', fontsize=15)
    
    else:
        plt.bar(data.index, data[metric_name], color='blue')

    average_value = data[metric_name].mean()
    plt.axhline(y=average_value, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {average_value:.2f}')
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.title(f'{model_name.upper()} - {metric_name} Visualization')
    plt.xlabel('Action Categories', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.tight_layout()

    # Save plot
    if metric_name == f'{bias_type.capitalize()} DR':
        plot_name = f'{bias_type}_dr.png'
    else:
        plot_name = f'{bias_type}_chi.png'

    plt.savefig(f'{output_folder}{plot_name}')
    plt.close()

def process_model_data(model_name: str):
    input_folder = f'data/{model_name}/'
    output_folder = f'visualizations/{model_name}/'
    

    combined_results = pd.DataFrame()

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))[['video_name', 'final']]
            df['category'] = df['video_name'].apply(extract_category_name)

            # Compute Dominance Ratio and Chi-Square
            results_df = compute_dominance_ratio_and_chi_square(df)

            # Determine Bias Type from Filename (gender or race)
            bias_type = file.split('_')[1].split('.')[0]
            output_folder = f'visualizations/{model_name}/{bias_type.capitalize()}/'
            os.makedirs(output_folder, exist_ok=True)
            results_df.columns = [f'{bias_type.capitalize()} Class', f'{bias_type.capitalize()} DR', f'{bias_type.capitalize()} Chi-Square']

            combined_results = pd.concat([combined_results, results_df], axis=1)

            # Save Dominance Ratio Plot
            plot_and_save(results_df, model_name, f'{bias_type.capitalize()} DR', bias_type, output_folder)

            # Save Chi-Square Plot
            plot_and_save(results_df, model_name, f'{bias_type.capitalize()} Chi-Square', bias_type, output_folder)

    # Save the combined results
    combined_results.index.name = 'Category'
    output_csv = f'data/{model_name}_scores.csv'
    combined_results.to_csv(output_csv)
    print(f"Scores saved to: {output_csv}")

def main():
    process_model_data('clip')
    process_model_data('llava')

if __name__ == "__main__":
    main()
