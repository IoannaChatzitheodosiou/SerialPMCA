import pandas as pd
import numpy as np
from dataclasses import dataclass

def get_max_values(df: pd.DataFrame) -> float:
    """
    Returns the maximum value in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to find the maximum value in.
    
    Returns:
    float: The maximum value in the DataFrame.
    """
    if df is None or df.empty:
        print("Error: The DataFrame is empty or None.")
        return None
    df.loc[0, -4:] = 0
    df.loc[1, -3:] = 0
    df.loc[2, -2:] = 0
    df.loc[3, -1:] = 0
    return df.to_numpy().max()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by filling NaN values with the mean of each column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame with NaN values filled.
    """
    if df is None or df.empty:
        print("Error: The DataFrame is empty or None.")
        return None
    df.loc[0, -4:] = 0
    df.loc[1, -3:] = 0
    df.loc[2, -2:] = 0
    df.loc[3, -1:] = 0
    df.fillna(df.mean(), inplace=True)
    #df_max = df.to_numpy().max()
    #normalize the data to the maximum value
    df = df /df.to_numpy().max()
    return df 

  

def calculate_euclidean_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Calculates the Euclidean distance between two DataFrames of the same shape.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.

    Returns:
    float: The Euclidean distance between the two DataFrames.
    """
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape.")
    return np.linalg.norm(df1.to_numpy() - df2.to_numpy())

def get_rms(df: pd.DataFrame) -> float:
    """
    Calculate the root mean square (RMS) value of all elements in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing numerical values.

    Returns:
        float: The RMS value of all elements in the DataFrame.
    """
    return np.sqrt(np.mean(df.to_numpy()**2))

@dataclass
class EmmissionRoi():
    min: float
    max: float

@dataclass
class ExcitationRoi():
    min: float
    max: float

def make_get_peak_shape(roi_peak_a: EmmissionRoi, roi_peak_b: EmmissionRoi):
    """
    Creates a function to calculate the peak shape based on two regions of interest (ROIs).

    Parameters:
    roi_peak_a (EmmissionRoi): The first ROI for peak A.
    roi_peak_b (EmmissionRoi): The second ROI for peak B.

    Returns:
    function: A function that takes a DataFrame and returns the peak shape.
    """
    return lambda df: get_peak_shape(df, roi_peak_a, roi_peak_b)


def get_peak_shape(df: pd.DataFrame, roi_peak_a: EmmissionRoi, roi_peak_b: EmmissionRoi) -> float:
    """
    Calculates the ratio of the summed values within two specified emission regions of interest (ROIs) in a DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame containing emission data, where columns represent emission wavelengths or channels.
        roi_peak_a (EmmissionRoi): The first emission region of interest, with 'min' and 'max' attributes specifying the column range.
        roi_peak_b (EmmissionRoi): The second emission region of interest, with 'min' and 'max' attributes specifying the column range.
    Returns:
        float: The ratio of the sum of values in roi_peak_b to the sum of values in roi_peak_a.
    Raises:
        ZeroDivisionError: If the sum of values in roi_peak_a is zero.
    """

    df_peak_a = df.loc[:, (df.columns >= roi_peak_a.min) & (df.columns <= roi_peak_a.max)]
    df_peak_b = df.loc[:, (df.columns >= roi_peak_b.min) & (df.columns <= roi_peak_b.max)]
    return df_peak_b.to_numpy().sum() / df_peak_a.to_numpy().sum()

def make_get_data_region(roi_emmision: EmmissionRoi, roi_excitation: ExcitationRoi):
    """
    Creates a function to extract a region of interest (ROI) from a DataFrame based on specified emission and excitation ROIs.

    Parameters:
    roi_emmision (EmmissionRoi): The emission ROI with 'min' and 'max' attributes.
    roi_excitation (ExcitationRoi): The excitation ROI with 'min' and 'max' attributes.

    Returns:
    function: A function that takes a DataFrame and returns the extracted ROI.
    """
    return lambda df: get_data_region(df, roi_emmision, roi_excitation) 

def get_data_region(df: pd.DataFrame, roi_emmision: EmmissionRoi, roi_excitation: ExcitationRoi) -> pd.DataFrame:
    """
    Extracts a region of interest (ROI) from a DataFrame based on specified minimum and maximum column values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    roi (EmmissionRoi): The region of interest with 'min' and 'max' attributes.

    Returns:
    pd.DataFrame: A DataFrame containing only the columns within the specified ROI.
    """
    return df.loc[(df.index >= roi_excitation.min) & (df.index <= roi_excitation.max), (df.columns >= roi_emmision.min) & (df.columns <= roi_emmision.max)]