import pandas as pd
import matplotlib.pyplot as plt
import logging


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def add_difference_column(df, col1, col2, new_col_name):
    """
    Adds a new column to the DataFrame that is the difference between two specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.
    new_col_name (str): The name of the new column to be created.

    Returns:
    pd.DataFrame: The DataFrame with the new difference column added.
    """
    logger.info(f"Calculating difference between {col1} and {col2}")
    df[new_col_name] = df[col1] - df[col2]
    
    # Preview relevant columns
    logger.info(
        "Preview of stress-related columns:\n%s",
        df[
            [
                "Stress_Level_Self_Report",
                "Stress_Level_Biosensor",
                "Stress_Difference",
            ]
        ].head()
    )
    return df

