import pandas as pd
import numpy as np

def read_data(file_path) -> pd.DataFrame:
    """
    Reads data from a CSV file and returns it as a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The data read from the CSV file.
    """
    data  = None
    try:
        data = np.genfromtxt(file_path, delimiter=',', skip_header=0)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    frame_columns = data[0, 1:]
    frame_index = data[1:, 0]
    data = data[1:, 1:]
    return pd.DataFrame(data, columns=frame_columns, index=frame_index)
