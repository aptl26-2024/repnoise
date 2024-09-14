# %%
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import re

# Read and parse the JSON Lines file
data = []
# with open('results_social.jsonl', 'r') as file:
# data_file = 'results_original_criminal.jsonl'
# data_file = 'results_repnoise_criminal.jsonl'
# data_file = 'results_original_social.jsonl'
data_file = 'results_repnoise_social.jsonl'
with open(data_file, 'r') as file:
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
df.head(5)
# %%

# Get unique combinations of ft_skip_split and jsonl_path
combinations = df.groupby(['ft_skip_split', 'jsonl_path'])

# duplicates = group[group.duplicated(subset=['epochs', 'lr'], keep=False)]
# print(duplicates)
# Create a heatmap for each combination
for (ft_skip, jsonl), group in combinations:
    # Pivot the data to create a 2D matrix
    # pivot_data = group.pivot(index='epochs', columns='lr', values='harmfulness_score')
    
    # # Create the heatmap
    # pivot_data *= 100
    # plt.figure(figsize=(5, 4))
    # sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.0f')
    
    # # Set title and labels
    # plt.title(f'Harmfulness Score Heatmap\nft_skip_split: {ft_skip}, jsonl_path: {jsonl}')
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Epochs')
    
    # # Show the plot
    # plt.tight_layout()
    # plt.show()
    if "harm" in jsonl:
        continue


    pivot_data = group.pivot(index='epochs', columns='lr', values='harmfulness_score')
    pivot_data *= 100
    plt.figure(figsize=(8, 6))

    # Create a custom colormap
    cmap = colors.LinearSegmentedColormap.from_list("custom", ["#FFF3E0", "#FF5722"])

    # Create the heatmap
    heatmap = sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap=cmap, 
        fmt='.0f',
        linewidths=0.5,
        # cbar_kws={'label': 'Harmfulness Score'},
        vmin=18,  # Set minimum value
        vmax=77
    )  # Set maximum value

    # Customize the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([pivot_data.min().min(), pivot_data.max().max()])
    # cbar.set_ticklabels(['Low', 'High'])

    # Set title and labels with custom styling
    # plt.title(f'Harmfulness Score Heatmap\nft_skip_split: {ft_skip}, jsonl_path: {jsonl.split("/")[-1]}', 
    #         fontsize=16, fontweight='bold', pad=20)
    plt.title(f'Accuracy on V {"with" if "original" not in data_file else "without"} RepNoise', 
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Learning Rate', fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel('Epochs', fontsize=14, fontweight='bold', labelpad=15)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add a subtle grid
    plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

    # Add a border around the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    # Adjust layout and display
    plt.tight_layout()
    import os
    output_dir = "heatmap"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{data_file}_{ft_skip}_jsonl_{jsonl.split('/')[-1].replace('.jsonl', '')}.svg"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as {filepath}")
    plt.show()
    plt.close()
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
