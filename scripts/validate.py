import os
import pandas as pd
import matplotlib.pyplot as plt

def load_files(model_name: str):
    folder_path = f'data/{model_name}/'
    files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    dataframes = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        bias_type = file.split('_')[1].split('.')[0]
        dataframes[bias_type] = df
    return dataframes

def process_groups(df: pd.DataFrame, bias_type: str):
    corrected_df = df.copy()
    invalid_samples = []

    group_key = df['video_name'].apply(lambda x: "_".join(x.split('_')[:3]))  # Extract the group identifier
    grouped = df.groupby(group_key)
    
    for group_name, group in grouped:
        dominant_class = group['final'].mode()[0]  # Find the most common label in the group
        for idx, row in group.iterrows():
            if row['final'] != dominant_class:
                # Save the original incorrect row to invalid samples
                invalid_samples.append(row.copy())
                # Update the row to the dominant class in the corrected DataFrame
                corrected_df.at[idx, 'final'] = dominant_class

    # Prepare invalid samples DataFrame
    invalid_df = pd.DataFrame(invalid_samples)

    return corrected_df, invalid_df

def save_results(model_name: str, bias_type: str, corrected_df: pd.DataFrame, invalid_df: pd.DataFrame):
    valid_folder = f'data/{model_name}/'
    invalid_folder = f'data/{model_name}/invalid/'
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(invalid_folder, exist_ok=True)

    # Remove the 'category' column from the corrected DataFrame before saving
    if 'category' in corrected_df.columns:
        corrected_df = corrected_df.drop(columns=['category'])

    valid_file = f'{valid_folder}{model_name}_valid{bias_type}.csv'
    invalid_file = f'{invalid_folder}{model_name}_invalid{bias_type}.csv'

    corrected_df.to_csv(valid_file, index=False)
    invalid_df.to_csv(invalid_file, index=False)
    print(f"Saved corrected file: {valid_file}")
    print(f"Saved invalid samples file: {invalid_file}")

def plot_invalid_samples(model_name: str, invalid_df: pd.DataFrame, corrected_df: pd.DataFrame, bias_type: str):
    if invalid_df.empty:
        print(f"No invalid samples for {model_name} - {bias_type}.")
        return

    # Count invalid samples per action category
    invalid_counts = invalid_df['category'].value_counts()
    total_counts = corrected_df['video_name'].apply(lambda x: x.split('_')[1]).value_counts()

    # Calculate percentage of invalid samples per category
    invalid_percentage = (invalid_counts / total_counts) * 100
    invalid_percentage = invalid_percentage.sort_index()  # Sort categories for consistent plotting

    plt.figure(figsize=(12, 6))
    plt.bar(invalid_percentage.index, invalid_percentage.values, color='blue')
    plt.title(f'{model_name.upper()} - Invalid Samples Percentage per Action Category ({bias_type})')
    plt.xlabel('Action Categories')
    plt.ylabel('Percentage of Invalid Samples')

    # Improve label visibility and readability
    plt.xticks(rotation=45, ha='right', fontsize=5)
    
    plt.tight_layout()
    
    # Save the plot
    output_folder = f'visualizations/{model_name}/'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f'{output_folder}{model_name}_{bias_type}_acc.png', dpi=300)  # Save with higher dpi for clarity
    plt.close()

def process_all_models():
    for model_name in ['clip', 'llava']:
        print(f"Processing model: {model_name}")
        dataframes = load_files(model_name)

        for bias_type, df in dataframes.items():
            print(f"Processing bias type: {bias_type}")
            df['category'] = df['video_name'].apply(lambda x: x.split('_')[1])  # Temporarily add category for calculation
            
            corrected_df, invalid_df = process_groups(df, bias_type)
            save_results(model_name, bias_type, corrected_df, invalid_df)
            plot_invalid_samples(model_name, invalid_df, corrected_df, bias_type)
            print(f"Finished processing bias type: {bias_type}")

        print(f"Finished processing model: {model_name}")

if __name__ == "__main__":
    process_all_models()
