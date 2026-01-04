import pandas as pd  # Import pandas for data manipulation. 
import matplotlib.pyplot as plt  # Import matplotlib for plotting. 
import seaborn as sns  # Import seaborn for statistical visualization. 

def process_and_clean_data(file_path):
    """
    Function to load, validate, and clean student health data.
    """
    # DATA LOADING 
    df = pd.read_csv(file_path)
    # We use len() to capture the total number of rows. 
    # Recording the length before and after filtering ensures data integrity and provides a clear audit trail.
    initial_count = len(df)
    
    # VALIDATION RULES 
    # Check if stress levels are within the logical biological range (0-10).
    valid_stress_biosensor = df['Stress_Level_Biosensor'].between(0, 10)
    valid_stress_self = df['Stress_Level_Self_Report'].between(0, 10)
    
    # Validate categorical data: Ensure gender is restricted to 'M' or 'F'.
    # The isin() function filters the data based on a list of allowed values. 
    # It returns "True" if the value exists in ['M', 'F'], ensuring categorical integrity.
    valid_gender = df['Gender'].isin(['M', 'F'])
    
    # DATA FILTERING 
    # Combine conditions using bitwise AND (&).
    valid_rows_mask = valid_stress_biosensor & valid_stress_self & valid_gender
    
    # Apply the Boolean Mask to the DataFrame. 
    # The mask contains 'True' for rows meeting our criteria and 'False' for those that don't.
    # We use .copy() to ensure 'df_cleaned' is a separate object in memory, preventing any accidental changes to the original 'df' dataset.
    df_cleaned = df[valid_rows_mask].copy()
    
    # REPORTING RESULTS
    # Calculate the difference to see how many rows failed the validation.
    removed_count = initial_count - len(df_cleaned)
    
    # Print summary statistics for transparency and verification of the cleaning process.
    print(f"Initial rows: {initial_count}")
    print(f"Rows removed (invalid data): {removed_count}")
    print(f"Cleaned dataset size: {len(df_cleaned)}")
    
    # OUTLIER DETECTION & VISUALIZATION
    # sns (Seaborn) is used for high-level statistical styling.
    # set_theme(style="whitegrid") provides a clean white background with grid lines. 
    # to enhance readability and make it easier to track values along the axes.
    sns.set_theme(style="whitegrid")
    
    # plt (Matplotlib) handles the window of the plot.
    # figsize=(10, 6) sets the dimensions of the figure in inches (Width=10, Height=6).
    # figsize=(10, 6) is chosen to maintain an optimal aspect ratio.
    # This size ensures that labels are legible and the data distribution is clearly visible without being cramped or distorted.
    plt.figure(figsize=(10, 6))
    
    # Using a Boxplot to visualize distribution and identify statistical outliers.
    sns.boxplot(data=df_cleaned[['Stress_Level_Biosensor', 'Stress_Level_Self_Report']])
    
    # Set the graph title to explain the content.
    plt.title('Statistical Distribution of Stress Levels')
    
    # Label the X-axis to identify the data groups.
    plt.xlabel('Stress Assessment Method')
    
    # Label the Y-axis to show the measurement scale.
    plt.ylabel('Measurement Scale (0-10)')
    
    # Display the final plot window.
    plt.show()
    
    # EXPORTING
    # Save the finalized cleaned data to a new CSV file for further group analysis.
    df_cleaned.to_csv('cleaned_student_health_data.csv', index=False)
    
    return df_cleaned

# RUNNING THE PROCESS
# Run the cleaning process on the CSV file.
final_data = process_and_clean_data('student_health_data.csv')
