import pandas as pd
import matplotlib.pyplot as plt
import logging


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

import os
print("CWD:", os.getcwd())

# Load cleaned data
logger.info("Loading cleaned data")
df = pd.read_csv("cleaned_student_health_data.csv")

# Log column names
logger.info("Listing column names")
for col in df.columns:
    logger.info(col)

# Create stress difference column
logger.info("Creating Stress_Difference column")
df["Stress_Difference"] = (
    df["Stress_Level_Self_Report"] - df["Stress_Level_Biosensor"]
)

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


