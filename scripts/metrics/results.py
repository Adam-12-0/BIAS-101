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

def load_and_prepare_data(model_name: str, bias_type: str):
    input_folder = f'data/{model_name}/'

    # Look for the file where the last part after '_' matches the bias type
    target_file = None
    for file in os.listdir(input_folder):
        if file.endswith('.csv') and file.split('_')[-1].replace('.csv', '') == bias_type:
            target_file = os.path.join(input_folder, file)
            break

    if not target_file:
        raise FileNotFoundError(f"No CSV file found in {input_folder} matching bias type '{bias_type}'.")

    data = pd.read_csv(target_file)
    data['category'] = data['video_name'].apply(lambda x: x.split('_')[1])

    category_counts = data.groupby(['category', 'final']).size().reset_index(name='count')
    category_counts['percentage'] = category_counts.groupby('category')['count'].transform(lambda x: (x / x.sum()))

    category_df = category_counts.pivot(index='category', columns='final', values='percentage').fillna(0)
    return category_df

def plot_stacked_bar_chart(model_name: str, bias_type: str, category_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 10))
    unique_subcategories = category_df.columns
    colors = {sub: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, sub in enumerate(unique_subcategories)}

    bottom = pd.Series([0] * len(category_df), index=category_df.index)
    for subcategory in unique_subcategories:
        ax.bar(category_df.index, category_df[subcategory], bottom=bottom, color=colors[subcategory], label=subcategory)
        bottom += category_df[subcategory]

    ax.set_title(f'{model_name.upper()} - {bias_type.capitalize()} Bias Distribution', fontsize=16)
    ax.set_xlabel('Action Categories', fontsize=12)
    ax.set_ylabel('Percentage of Videos (0 to 1)', fontsize=12)
    ax.set_xticks(range(len(category_df.index)))
    ax.set_xticklabels(category_df.index, rotation=45, ha='right', fontsize=5)
    ax.legend(loc='upper left', fontsize=15)
    plt.tight_layout()

    output_folder = f'visualizations/{model_name}/{bias_type.capitalize()}/'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f'{output_folder}/{bias_type}_results.png')
    plt.close()

def main():
    for model_name in [folder for folder in os.listdir('data') if os.path.isdir(os.path.join('data', folder))]:
        input_folder = os.path.join('data', model_name)

        # Collect bias types by looking at the *last* part of the file name before ".csv"
        bias_types = set()
        for file in os.listdir(input_folder):
            if file.endswith('.csv'):
                parts = file.replace('.csv', '').split('_')
                bias_type = parts[-1]  # Safely extract the last part
                bias_types.add(bias_type)

        for bias_type in bias_types:
            category_df = load_and_prepare_data(model_name, bias_type)
            plot_stacked_bar_chart(model_name, bias_type, category_df)

if __name__ == "__main__":
    main()
