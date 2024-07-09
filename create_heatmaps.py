# %%
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Read and parse the JSON Lines file
data = []
# with open('results_social.jsonl', 'r') as file:
with open('results_base_social.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Extract relevant information
df = pd.DataFrame([
    {
        'ft_skip_split': re.search(r'ft-skip_split(\d+)', d['model']).group(1) if re.search(r'ft-skip_split(\d+)', d['model']) else 'unknown',
        'epochs': int(re.search(r'epoch(\d+)', d['model']).group(1)) if re.search(r'epoch(\d+)', d['model']) else 0,
        # 'lr': float(re.search(r'lr([\d.e-]+)', d['model']).group(1)) if re.search(r'lr([\d.e-]+)', d['model']) else 0.0,
        # 'lr': float(re.search(r'lr([\d.e-]+?(?:e-?[\d]+)?)', d['model']).group(1)) if re.search(r'lr([\d.e-]+?(?:e-?[\d]+)?)', d['model']) else 0.0,
        'lr': float(re.search(r'lr([\d.]+e-?\d+)', d['model']).group(1)) if re.search(r'lr([\d.]+e-?\d+)', d['model']) else 0.0,
        'jsonl_path': d['jsonl_path'],
        'harmfulness_score': d['harmfulness_score']
    }
    for d in data
])

# display(df)

# %%

# Get unique combinations of ft_skip_split and jsonl_path
combinations = df.groupby(['ft_skip_split', 'jsonl_path'])

# duplicates = group[group.duplicated(subset=['epochs', 'lr'], keep=False)]
# print(duplicates)
# Create a heatmap for each combination
for (ft_skip, jsonl), group in combinations:
    # Pivot the data to create a 2D matrix
    pivot_data = group.pivot(index='epochs', columns='lr', values='harmfulness_score')
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.2f')
    
    # Set title and labels
    plt.title(f'Harmfulness Score Heatmap\nft_skip_split: {ft_skip}, jsonl_path: {jsonl}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Epochs')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# %%
# Create heatmaps for each ft_skip_split and jsonl_path combination
for ft_skip_split in df['ft_skip_split'].unique():
    for jsonl_path in df['jsonl_path'].unique():
        subset = df[(df['ft_skip_split'] == ft_skip_split) & (df['jsonl_path'] == jsonl_path)]
        if not subset.empty:
            pivot = subset.pivot(index='epochs', columns='lr', values='harmfulness_score')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.2f')
            plt.title(f'Harmfulness Score Heatmap\nft_skip_split: {ft_skip_split}, jsonl_path: {jsonl_path}')
            plt.xlabel('Learning Rate')
            plt.ylabel('Epochs')
            plt.savefig(f'heatmap_{ft_skip_split}_{jsonl_path.split("/")[-1]}.png')
            plt.close()
        else:
            print(f"No data for ft_skip_split: {ft_skip_split}, jsonl_path: {jsonl_path}")

print('Heatmaps have been generated and saved as PNG files.')

# %%
