import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_data_logic(df):
    """
    Logic for validating and filtering data.
    """
    # Check if stress levels are within the logical biological range (0-10).
    valid_stress_biosensor = df['Stress_Level_Biosensor'].between(0, 10)
    valid_stress_self = df['Stress_Level_Self_Report'].between(0, 10)
    
    # Convert gender strings to uppercase.
    # This ensures consistency even if the input data has mixed casing.
    gender_upper = df['Gender'].str.upper()
    valid_gender = gender_upper.isin(['M', 'F'])

    # Combine all rules into a single mask.
    valid_rows_mask = valid_stress_biosensor & valid_stress_self & valid_gender
    
    return df[valid_rows_mask].copy()


def process_and_clean_data(file_path):
    """
    Coordinates loading, cleaning, visualization, and export.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    initial_count = len(df)

    # Apply cleaning logic
    df_cleaned = clean_data_logic(df)
    removed_count = initial_count - len(df_cleaned)

    # Log results instead of printing
    logger.info(f"Initial rows: {initial_count}")
    logger.info(f"Rows removed: {removed_count}")
    logger.info(f"Cleaned dataset size: {len(df_cleaned)}")

    # Visualize stress distributions
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Selection for plotting.
    plot_cols = ['Stress_Level_Biosensor', 'Stress_Level_Self_Report']
    sns.boxplot(data=df_cleaned[plot_cols])

    plt.title('Statistical Distribution of Stress Levels')
    plt.xlabel('Stress Assessment Method')
    plt.ylabel('Measurement Scale (0-10)')
    plt.show()

    # Export cleaned data
    output_filename = 'cleaned_student_health_data.csv'
    df_cleaned.to_csv(output_filename, index=False)
    
    logger.info(f"Process complete. Data saved to: {output_filename}")
    
    return df_cleaned


if __name__ == "__main__":
    process_and_clean_data('student_health_data.csv')
