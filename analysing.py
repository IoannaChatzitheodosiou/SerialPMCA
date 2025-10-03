import pandas as pd
import numpy as np
from typing import Dict, Callable, List
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from plotting import Plotter

class DataAnalyzer:
    def __init__(self, data: Dict[str, pd.DataFrame], get_max_values: Callable, preprocess: Callable, 
                 get_peak_shape: Callable, get_rms: Callable, 
                 get_euclidean_distance: Callable, get_data_region: Callable,
                 plotter: Plotter ,groups: dict = None):
        """
        Initialize the DataAnalyzer.
        data: Dictionary of DataFrames (or lists of DataFrames) keyed by sample name.
        groups: Optional dictionary of groupings (e.g., from groups.yaml) for grouped analysis.
        Other arguments are processing and plotting functions.
        """
        self._data = data
        self._preprocess = preprocess
        self._get_peak_shape = get_peak_shape
        self._get_rms = get_rms
        self._get_euclidean_distance = get_euclidean_distance
        self._get_data_region = get_data_region
        self._get_max_values = get_max_values
        self._plotter = plotter
        self._groups = groups  # Store group information if provided
        self._max_values = {}  # Dictionary to store {name: max} for each DataFrame

    def analyze(self):
        for key, df in self._data.items():
            print(f"Analyzing {key}...")
            self._data[key] = self._preprocess(df)
            if self._data[key] is None:
                print(f"Warning: DataFrame for {key} is None after preprocessing. Skipping.")
                continue
            self._data[key] = self._get_data_region(self._data[key])
            print(f"Completed preprocessing for {key}.")
        self.max_values()  
        self._plot_max_values_method()  
        #self._plot_euclidean_heatmap()
        self._plot_eem_features()
        self._group_peak_shapes()

    def max_values(self):
        for key, df in self._data.items():
            self._max_values[key] = self._get_max_values(df)
        print("\n Max values for each DataFrame:")
        for key, max_val in self._max_values.items():
            print(f"{key}: {max_val}")

    def _plot_max_values_method(self):
        self._plotter.plot_max_values(self._max_values)

    def _plot_euclidean_heatmap(self):
        # Build a pairwise distance matrix between all DataFrames in self._data
        data = {}
        for key1, df1 in self._data.items():
            data1 = {}
            for key2, df2 in self._data.items():
                # Calculate the Euclidean distance between df1 and df2
                data1[key2] = self._get_euclidean_distance(df1, df2)
            data[key1] = data1  # Store the row for key1
        # Convert the nested dictionary to a DataFrame and plot the heatmap
        self._plotter.plot_euclidean_heatmap(pd.DataFrame(data))


    def _plot_eem_features(self):
        # Prepare lists for plotting peak shape vs RMS for each sample
        x = []  # List to store peak shape values
        y = []  # List to store RMS values
        labels = []  # List to store sample keys (labels)
        for key, df in self._data.items():
            # Calculate peak shape and RMS for each DataFrame
            x.append(self._get_peak_shape(df))
            y.append(self._get_rms(df))
            labels.append(key)
        # Call the plotting function with the computed values and labels
        self._plotter.plot_eem_features(x, y, labels)

    def _group_peak_shapes(self):
        @dataclass
        class peak_plot_help:
            peak: List
            rms: List
            labels: List
            peak_err: List
            rms_err: List

            def populate(self, peak, rms, label):
                self.peak.append(np.mean(peak))
                self.rms.append(np.mean(rms))
                self.peak_err.append(np.std(peak))
                self.rms_err.append(np.std(rms))
                self.labels.append(label)

        # Prepare lists for averaged peak shape vs RMS for each group
        subgroup_plot = peak_plot_help([], [], [], [], [])
        group_plot = peak_plot_help([], [], [], [], [])
        for group, subgroup in self._groups.items():
            peak_shapes_group = []
            rms_values_group = []
            for row, ids in subgroup.items():
                peak_shapes_sub = []
                rms_values_sub = []
                for id in ids:
                    # Calculate peak shape and RMS for each DataFrame in the group row
                    peak_shapes_sub.append(self._get_peak_shape(self._data[id]))
                    rms_values_sub.append(self._get_rms(self._data[id]))
                    peak_shapes_group.append(self._get_peak_shape(self._data[id]))
                    rms_values_group.append(self._get_rms(self._data[id]))
                if peak_shapes_sub and rms_values_sub:
                    subgroup_plot.populate(peak_shapes_sub, rms_values_sub, f'{group}/{row}')
            if peak_shapes_group and rms_values_group:
                group_plot.populate(peak_shapes_group, rms_values_group, group)
        # Call the plotting function with the averaged values and group labels
        # You can use x_err and y_err for error bars in your plotting function
        self._plotter.plot_eem_features(subgroup_plot.peak, subgroup_plot.rms, subgroup_plot.labels)
        self._plotter.plot_eem_features(subgroup_plot.peak, subgroup_plot.rms, subgroup_plot.labels, 
                                    subgroup_plot.peak_err, subgroup_plot.rms_err)
        self._plotter.plot_eem_features(group_plot.peak, group_plot.rms, group_plot.labels)
        self._plotter.plot_eem_features(group_plot.peak, group_plot.rms, group_plot.labels,
                                    group_plot.peak_err, group_plot.rms_err)
        
