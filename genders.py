import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import logging

# Setup logger for the example
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import logging

# Setup a basic logger for demonstration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import logging

# Setup logger for the example
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import logging

# Setup a basic logger for demonstration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardize_categorical_column(
    df: pd.DataFrame, 
    candidate_cols: list[str], 
    category_mapping: dict[str, list[str]], 
    output_col_name: str = 'standardized_col'
) -> pd.DataFrame | None:
    """
    Generalizes column standardization by accepting dynamic column names and mapping rules.

    Args:
        df (pd.DataFrame): The input DataFrame.
        candidate_cols (list[str]): A list of potential column names to search for (case-insensitive).
        category_mapping (dict[str, list[str]]): A dictionary mapping the desired output value to a list of possible input variations.
    Returns:
        pd.DataFrame: Cleaned DataFrame with the new standardized column, 
                      or None if the target column is not found.
    """
    new_df = df.copy()

    # Identify the target column dynamically
    candidates_lower = {c.lower() for c in candidate_cols}
    
    # Find the first column in the DF that matches one of the candidates
    target_col = next((col for col in df.columns if col.lower() in candidates_lower), None)

    if not target_col:
        logger.error(f"None of the candidate columns {candidate_cols} found in the dataset.")
        return None

    logger.info(f"Found target column: '{target_col}'")

    # Prepare the lookup dictionary
    lookup_map = {}
    for standard_val, variations in category_mapping.items():
        for var in variations:
            # Convert variation to string and lowercase to ensure robust matching
            lookup_map[str(var).lower().strip()] = standard_val

    raw_series = new_df[target_col].astype(str).str.strip().str.lower()
    new_df[output_col_name] = raw_series.map(lookup_map)

    # Log results
    counts = new_df[output_col_name].value_counts()
    logger.info(f"Standardization complete for '{output_col_name}'. Counts: {counts.to_dict()}")

    # Drop rows where the value could not be standardized
    new_df = new_df.dropna(subset=[output_col_name])

    return new_df

def get_gender_series(df: pd.DataFrame, target_col: str, gender_col: str) -> tuple:
    """
    Splits the data into Male ('M') and Female ('F') Series based on the target column.
    """
    # 1. Validation: specific columns must exist
    if target_col not in df.columns or gender_col not in df.columns:
        logger.error(f"Missing columns: {target_col} or {gender_col}")
        return None, None

    # 2. Extract Series using boolean indexing (faster and cleaner than groupby)
    male_series = df.loc[df[gender_col] == 'M', target_col]
    female_series = df.loc[df[gender_col] == 'F', target_col]

    # 3. Validation: Ensure both groups have data
    if male_series.empty or female_series.empty:
        logger.warning("One of the gender groups is empty.")
        return None, None

    return male_series, female_series
def get_gender_series(df: pd.DataFrame, target_col: str, gender_col: str) -> tuple:
    """
    Splits the data into Male ('M') and Female ('F') Series based on the target column.
    """
    # 1. Validation: specific columns must exist
    if target_col not in df.columns or gender_col not in df.columns:
        logger.error(f"Missing columns: {target_col} or {gender_col}")
        return None, None

    # 2. Extract Series using boolean indexing (faster and cleaner than groupby)
    male_series = df.loc[df[gender_col] == 'M', target_col]
    female_series = df.loc[df[gender_col] == 'F', target_col]

    # 3. Validation: Ensure both groups have data
    if male_series.empty or female_series.empty:
        logger.warning("One of the gender groups is empty.")
        return None, None

    return male_series, female_series