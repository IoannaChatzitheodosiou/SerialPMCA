from analysing import DataAnalyzer
from processing import preprocess_data, get_max_values,get_peak_shape, get_rms, calculate_euclidean_distance, make_get_peak_shape, EmmissionRoi, ExcitationRoi, make_get_data_region
from read_data import read_data
from plotting import Plotter
import numpy as np
import yaml

def main():

    # Load the data.yaml file and read each file into a DataFrame
    data = {}
    with open('data.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
        for key, file in yaml_data.items():
            # Read each file path from the YAML into a DataFrame and store in the data dict
            data[key] = read_data(file)

    # Load the groups.yaml file and build a dictionary of grouped DataFrames
    groups_data = {}
    with open('groups.yaml', 'r') as f:
        groups_data = yaml.safe_load(f)

    # Create the DataAnalyzer object with all required processing and plotting functions
    analyzer = DataAnalyzer(
        data=data,  # The dictionary of DataFrames loaded from data.yaml
        get_max_values=get_max_values,  # Function to get max values from each DataFrame
        preprocess=preprocess_data,  # Function to preprocess each DataFrame
        get_peak_shape=make_get_peak_shape(EmmissionRoi(510, 554), EmmissionRoi(556, 650)),  # Function to calculate peak shape
        get_rms=get_rms,  # Function to calculate RMS
        get_euclidean_distance=calculate_euclidean_distance,  # Function to calculate Euclidean distance
        get_data_region=make_get_data_region(EmmissionRoi(520, 610), ExcitationRoi(440, 500)),  # Function to extract a region of interest
        plotter=Plotter(),  # The Plotter object with plotting methods
        groups=groups_data,  # The grouped DataFrames for group-based analysis
        
    )



    # Run the analysis pipeline
    analyzer.analyze()

    # # Plot the grouped peak shapes (averages and std)
    # if hasattr(analyzer, '_group_peak_shapes'):
    #     x, y, labels, x_err, y_err = analyzer._group_peak_shapes()
    #     plot_eem_features(x, y, labels, x_err, y_err)
    #     # Also plot the grouped heatmap using the same labels and y values as a demo
    #     plot_grouped_heatmap(np.column_stack((x, y)), labels)

if __name__ == "__main__":
    main()