import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Logger 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def load_data(path):
    return pd.read_csv(path)


def add_stress_difference(df):
    if "Stress_Difference" not in df.columns:
        logger.info("Creating Stress_Difference column")
        df["Stress_Difference"] = df["Stress_Level_Self_Report"] - df["Stress_Level_Biosensor"]
    return df


def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    logger.info("Plotting correlation heatmap for numeric columns")
    plt.figure(figsize=(12, 9))
    sns.heatmap(df[numeric_cols].corr(), center=0)
    plt.title("Correlation Matrix (Numeric Variables)")
    plt.tight_layout()
    plt.show()

###############################################################################
def main():
    logger.info("Loading cleaned dataset")
    df = load_data("cleaned_student_health_data.csv")

    df = add_stress_difference(df)

    plot_correlation_heatmap(df)

    logger.info("Done")


if __name__ == "__main__":
    main()
