import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("cleaned_student_health_data.csv")

# Print column names
for col in df.columns:
    print(col)

# Create stress difference column
df["Stress_Difference"] = (
    df["Stress_Level_Self_Report"] - df["Stress_Level_Biosensor"]
)

# Preview relevant columns
print(df[[
    "Stress_Level_Self_Report",
    "Stress_Level_Biosensor",
    "Stress_Difference"
]].head())

# Scatter plot: self-reported vs. measured stress
plt.figure(figsize=(7, 7))
plt.scatter(
    df["Stress_Level_Self_Report"],
    df["Stress_Level_Biosensor"],
    alpha=0.6
)

# Perfect agreement line (y = x)
min_val = min(
    df["Stress_Level_Self_Report"].min(),
    df["Stress_Level_Biosensor"].min()
)
max_val = max(
    df["Stress_Level_Self_Report"].max(),
    df["Stress_Level_Biosensor"].max()
)
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Self-reported Stress Level")
plt.ylabel("Measured Stress Level (Biosensor)")
plt.title("Self-reported vs. Measured Stress Levels")

plt.tight_layout()
plt.show()