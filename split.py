import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

def standardize_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the gender/sex column, standardizes values to 'Male'/'Female',
    and returns a cleaned DataFrame.
    
    Handles:
    - Column names: 'gender', 'sex' (case insensitive).
    - Values: 'm', 'male', 'man', '1' -> 'Male'.
    - Values: 'f', 'female', 'woman', '0' -> 'Female'.
    """
    df_clean = df.copy()
    
    # 1. Identify the gender column dynamically
    gender_col = next((col for col in df.columns if col.lower() in ['gender', 'sex']), None)
    
    if not gender_col:
        logger.error("No 'gender' or 'sex' column found in the dataset.")
        return None

    logger.info(f"Found gender column: '{gender_col}'")

    # 2. Normalize to lowercase and strip whitespace for robust matching
    # Convert to string first to handle numeric 1/0 inputs gracefully
    s = df_clean[gender_col].astype(str).str.strip().str.lower()

    # 3. Define conditions
    conditions = [
        s.isin(['m', 'male', 'man', '1']),
        s.isin(['f', 'female', 'woman', '0'])
    ]
    choices = ['Male', 'Female']

    # 4. Create a new standardized column
    df_clean['standardized_gender'] = np.select(conditions, choices, default=np.nan)

    # Log how many were classified
    counts = df_clean['standardized_gender'].value_counts()
    logger.info(f"Gender standardization complete. Counts: {counts.to_dict()}")

    # Drop rows where gender could not be determined
    df_clean = df_clean.dropna(subset=['standardized_gender'])
    
    return df_clean


def get_gender_series(df: pd.DataFrame, target_col: str, gender_col: str, group_keys: dict) -> tuple:
    """
    Splits the data into two Series (groups) using groupby, based on specific column names.
    
    Args:
        df (pd.DataFrame): The dataset containing the data.
        target_col (str): The name of the column containing the metric to analyze (e.g., 'pressure_diff').
        gender_col (str): The name of the column containing gender labels (e.g., 'standardized_gender').
        group_keys (dict): A dictionary mapping logical groups to actual values in the dataframe.
                           Example: {'male': 'Male', 'female': 'Female'}
                           
    Returns:
        tuple: (male_series, female_series) - Two pandas Series objects containing the values of target_col.
               Returns (None, None) if a group key is missing in the data.
    """
    # 1. Validation: Ensure columns exist in the DataFrame
    if target_col not in df.columns or gender_col not in df.columns:
        logger.error(f"Columns {target_col} or {gender_col} not found in DataFrame.")
        return None, None

    # 2. Create a GroupBy object: Group by gender, but select only the target column (returns SeriesGroupBy)
    # This is more memory efficient than grouping the entire dataframe if we only need one column.
    grouped = df.groupby(gender_col)[target_col]
    
    try:
        # 3. Extract the specific Series for each group using the provided keys
        # We use .get_group() to retrieve the specific Series for 'Male' and 'Female'
        male_series = grouped.get_group(group_keys['male'])
        female_series = grouped.get_group(group_keys['female'])
        
        logger.info(f"Successfully split groups: Male count={len(male_series)}, Female count={len(female_series)}")
        return male_series, female_series

    except KeyError as e:
        # This handles cases where one gender might be completely missing from the data
        logger.error(f"Group key not found in the dataset: {e}. Check standardization.")
        return None, None