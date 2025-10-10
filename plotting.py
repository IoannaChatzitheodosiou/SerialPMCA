import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Dict
import yaml



class Plotter:

    def __init__(self, colours_file: str, groups_config: Dict):
        # Load colors from the YAML file
        with open(colours_file, 'r') as f:
            self._colours = yaml.safe_load(f)
        self._groups_config = groups_config

    def get_colour(self, label: str) -> str:
        """
        Returns the color associated with the given label from the loaded colors dictionary.
        If the label is not found, returns a default color.
        """
        label = label.split('/')[0]
        for group, subgroups in self._groups_config.items():
            if label == group:
                return self._colours.get(label,"#000000")
            for subgroup, measurements in subgroups.items():
                if label == subgroup:
                    return self._colours.get(subgroup, "#000000")
                for measurement in measurements:
                    if label.split('_')[0] == measurement.split('_')[0]:
                        return self._colours.get(group, "#000000")
        return "#000000"

    def plot_max_values(self, max_values: dict):
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
        colors = [self.get_colour(label) for label in labels]


        # Create scatter plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=colors)
        plt.xticks(rotation=90, ha='right')
        plt.ylabel('Maximum Value')
        plt.title('Maximum Values for Each DataFrame')
        plt.tight_layout()
        plt.savefig('max_values.pdf') # Save plot to file
        plt.show()
        plt.close()

    def plot_eem_features(self, x: list, y: list, labels: list, x_err: list = None, y_err: list = None, cmap: str = 'tab20'):
        """
        Plots peak shapes with optional error bars and group coloring.
        Adds arrows between consecutive generations if group labels start with 'gen'.
        """
        # Extract group from label (assumes label format 'group_row')
        groups = [str(label).split('_')[0] for label in labels]
        df = pd.DataFrame({'x': x, 'y': y, 'label': labels, 'group': groups})
        # Assign a color to each group using the specified colormap
        unique_groups = df['group'].unique()
        group_color_map = {group: self.get_colour(group) for group in unique_groups}
        if x_err is not None and y_err is not None:
            # Plot error bars for each group
            for g in unique_groups:
                idx = df['group'] == g
                plt.errorbar(df.loc[idx, 'x'], df.loc[idx, 'y'],
                            xerr=np.array(x_err)[idx], yerr=np.array(y_err)[idx],
                            fmt='o', ecolor=group_color_map[g], color=group_color_map[g], capsize=5, linestyle='None', label=g)
            plt.legend(title='Group')
        else:
            # Plot scatter plot with group coloring
            sns.scatterplot(data=df, x='x', y='y', hue='group', palette=group_color_map)
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
        plt.tight_layout()
        plt.savefig('peak_shapes.pdf')  # Save plot to file
        plt.show()
        plt.close()

    def plot_euclidean_heatmap(self, data: pd.DataFrame):
        """
        Plots a heatmap of the provided data using Euclidean distances.
        """
        sns.heatmap(pd.DataFrame(data), annot=True, cmap='viridis')
        plt.tight_layout()
        plt.savefig('euclidean_map.pdf')  # Save plot to file
        plt.show()
        plt.close()

    def plot_grouped_heatmap(self, x: list, y: list, labels: list):
        """
        Plots a heatmap where each cell is the mean of the group (grouped by the prefix before '_').
        data: pd.DataFrame or np.ndarray (samples x features)
        labels: list of sample labels, format 'group_row' or similar
        """
        # Extract group from groups.yaml (assumes groups.yaml contains a mapping from label to group)
        # Extract group from label (assumes label format 'group_row')
        groups = [str(label).split('_')[0] for label in labels]
        df = pd.DataFrame({'x': x, 'y': y, 'label': labels, 'group': groups})
        # Group by 'group' and average all columns except 'group'
        group_means = df.groupby('group').mean(numeric_only=True)

        # # Compute pairwise Euclidean distances between group means
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
        plt.savefig('grouped_heatmap.pdf')  # Save plot to file
        plt.show()
        plt.close()