import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import re
import yaml

def plot_max_values(max_values: dict):
    """
    Plots the maximum values from a dictionary where keys are labels and values are the max values.
    """
    # Check if the dictionary is empty
    if not max_values:
        print("No maximum values to plot.")
        return

    # Prepare labels and values for plotting
    labels = [str(label) for label in max_values.keys()]  # Ensure labels are strings
    values = list(max_values.values())

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=values, palette='viridis')
    plt.xticks(rotation=90, ha='right')
    plt.ylabel('Maximum Value')
    plt.title('Maximum Values for Each DataFrame')
    plt.tight_layout()
    plt.show()
    plt.savefig('max_values.png')  # Save plot to file
    plt.close()

def plot_eem_features(x: list, y: list, labels: list, x_err: list = None, y_err: list = None, cmap: str = 'tab20'):
    """
    Plots peak shapes with optional error bars and group coloring.
    Adds arrows between consecutive generations if group labels start with 'gen'.
    """
    # Extract group from label (assumes label format 'group_row')
    groups = [str(label).split('_')[0] for label in labels]
    df = pd.DataFrame({'x': x, 'y': y, 'label': labels, 'group': groups})

    if x_err is not None and y_err is not None:
        # Assign a color to each group using the specified colormap
        unique_groups = df['group'].unique()
        palette = sns.color_palette("tab20", n_colors=len(unique_groups))
        group_color_map = {g: palette[i] for i, g in enumerate(unique_groups)}
        # Plot error bars for each group
        for g in unique_groups:
            idx = df['group'] == g
            plt.errorbar(df.loc[idx, 'x'], df.loc[idx, 'y'],
                         xerr=np.array(x_err)[idx], yerr=np.array(y_err)[idx],
                         fmt='o', ecolor=group_color_map[g], color=group_color_map[g], capsize=5, linestyle='None', label=g)
        plt.legend(title='Group')
    else:
        # Plot scatter plot with group coloring
        sns.scatterplot(data=df, x='x', y='y', hue='group')
        # # Annotate each point with its label
        # for xi, yi, label in zip(df['x'], df['y'], df['label']):
        #     plt.text(xi, yi, label, fontsize=9, ha='right', va='bottom')

    # Only add arrows if there are any group labels starting with 'gen' (case-insensitive)
    gen_groups = [g for g in groups if g.lower().startswith('gen')]
    if gen_groups:
        # Add arrows between consecutive generations (assumes group names like 'Gen1', 'Gen2', ...)
        # Find all unique generations in order
        generations = sorted(set(gen_groups), key=lambda g: (g.lower().replace('gen','').zfill(3) if g.lower().startswith('gen') else g))
        gen_points = {}
        for gen in generations:
            # Find indices for this generation
            idx = [i for i, g in enumerate(groups) if g == gen]
            if idx:
                # Average x and y for this generation
                x_avg = np.mean([x[i] for i in idx])
                y_avg = np.mean([y[i] for i in idx])
                gen_points[gen] = (x_avg, y_avg)

        # # Draw arrows between consecutive generations
        # gen_list = [g for g in generations if g in gen_points]
        # for i in range(len(gen_list)-1):
        #     x_start, y_start = gen_points[gen_list[i]]
        #     x_end, y_end = gen_points[gen_list[i+1]]
        #     plt.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
        #                  arrowprops=dict(arrowstyle='->', color='black', lw=1))
    plt.xlabel('Peak Shape')
    plt.ylabel('RMS')
    plt.show()
    plt.savefig('peak_shapes.png')  # Save plot to file
    plt.close()

def plot_euclidean_heatmap(data: pd.DataFrame):
    """
    Plots a heatmap of the provided data using Euclidean distances.
    """
    sns.heatmap(pd.DataFrame(data), annot=True, cmap='viridis')
    plt.show()
    plt.savefig('euclidean_map.png')  # Save plot to file
    plt.close()

def plot_grouped_heatmap(data: pd.DataFrame, labels: list):
    """
    Plots a heatmap where each cell is the mean of the group (grouped by the prefix before '_').
    data: pd.DataFrame or np.ndarray (samples x features)
    labels: list of sample labels, format 'group_row' or similar
    """
    # Extract group from groups.yaml (assumes groups.yaml contains a mapping from label to group)
    with open('groups.yaml', 'r') as f:
        group_map = yaml.safe_load(f)
    # Map each label to its group
    groups = [group_map.get(label, label) for label in labels]
    df = pd.DataFrame(data)
    df['group'] = groups
    # Group by 'group' and average all columns except 'group'
    group_means = df.groupby('group').mean(numeric_only=True)
    print (df.groupby)

    # # Compute pairwise Euclidean distances between group means
    # def natural_key(s):
    #     # Split string into list of strings and integers for natural sorting
    #     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    group_names = sorted(group_means.index.tolist())
    # Reorder group_means to match sorted group_names
    group_means_sorted = group_means.loc[group_names]
    distances = cdist(group_means_sorted.values, group_means_sorted.values, metric='euclidean')
    dist_df = pd.DataFrame(distances, index=group_names, columns=group_names)

    # Plot heatmap of distances
    sns.heatmap(dist_df, annot=True, cmap='viridis')
    plt.title('Euclidean Distance Between Groups')
    plt.ylabel('Group')
    plt.xlabel('Group')
    plt.tight_layout()
    plt.show()
    plt.savefig('grouped_heatmap.png')  # Save plot to file
    plt.close()