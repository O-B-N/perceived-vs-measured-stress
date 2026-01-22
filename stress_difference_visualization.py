import pandas as pd
import matplotlib.pyplot as plt


# Load dataset from CSV file
def load_data(path):
    return pd.read_csv(path)


# Add derived variable: difference between self-report and biosensor
def add_stress_difference(df):
    if "Stress_Difference" not in df.columns:
        df["Stress_Difference"] = (
            df["Stress_Level_Self_Report"]
            - df["Stress_Level_Biosensor"]
        )
    return df


# Scatter plot: self-reported vs biosensor stress by gender
def plot_scatter_by_gender(df):
    df_m = df[df["Gender"] == "M"]
    df_f = df[df["Gender"] == "F"]

    plt.figure(figsize=(7, 7))

    plt.scatter(
        df_m["Stress_Level_Self_Report"],
        df_m["Stress_Level_Biosensor"],
        alpha=0.6,
        label="Male"
    )

    plt.scatter(
        df_f["Stress_Level_Self_Report"],
        df_f["Stress_Level_Biosensor"],
        alpha=0.6,
        label="Female"
    )

    min_val = min(
        df["Stress_Level_Self_Report"].min(),
        df["Stress_Level_Biosensor"].min()
    )
    max_val = max(
        df["Stress_Level_Self_Report"].max(),
        df["Stress_Level_Biosensor"].max()
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Perfect agreement (y = x)"
    )

    plt.xlabel("Self-reported Stress Level")
    plt.ylabel("Measured Stress Level (Biosensor)")
    plt.title("Self-reported vs. Measured Stress Levels by Gender")

    plt.legend()
    plt.tight_layout()
    plt.show()



# Boxplot: stress difference by gender
def plot_stress_difference_boxplot(df):
    m_diff = df.loc[df["Gender"] == "M", "Stress_Difference"].dropna()
    f_diff = df.loc[df["Gender"] == "F", "Stress_Difference"].dropna()

    plt.figure(figsize=(7, 5))

    plt.boxplot([m_diff, f_diff], tick_labels=["Male", "Female"])

    plt.axhline(0, linestyle="--")
    
    plt.ylabel("Stress Difference (Self-report - Biosensor)")
    plt.title("Stress Difference by Gender")

    plt.tight_layout()
    plt.show()



# Histograms: distribution of stress difference by gender
def plot_stress_difference_histograms(df):
    m_diff = df.loc[df["Gender"] == "M", "Stress_Difference"].dropna()
    f_diff = df.loc[df["Gender"] == "F", "Stress_Difference"].dropna()

    plt.figure(figsize=(7, 5))

    plt.hist(m_diff, bins=30, alpha=0.6, label="Male")
    plt.hist(f_diff, bins=30, alpha=0.6, label="Female")

    plt.axvline(0, linestyle="--")

    plt.xlabel("Stress Difference (Self-report - Biosensor)")
    plt.ylabel("Count")
    plt.title("Distribution of Stress Difference by Gender")

    plt.legend()
    plt.tight_layout()
    plt.show()

###########################################################################
def main():
    df = load_data("cleaned_student_health_data.csv")

    df = add_stress_difference(df)

    plot_scatter_by_gender(df)
    plot_stress_difference_boxplot(df)
    plot_stress_difference_histograms(df)

if __name__ == "__main__":
    main()
