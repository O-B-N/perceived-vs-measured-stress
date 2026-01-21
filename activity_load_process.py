def process_and_visualize_load(df):
    activity_mapping = {
        'LOW': 1.0,
        'MODERATE': 3.75,
        'HIGH': 7.0
    }
    
    df['Physical_Activity_Cleaned'] = df['Physical_Activity'].str.strip().str.upper()
    
    df['Physical_Activity_Hours'] = df['Physical_Activity_Cleaned'].map(activity_mapping).fillna(0)
    
    df['Combined_Weekly_Load'] = df['Study_Hours'] + df['Project_Hours'] + df['Physical_Activity_Hours']
    
    # 3. Visualization: Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='Combined_Weekly_Load', 
        y='Stress_Level_Self_Report', 
        hue='Gender', 
        alpha=0.6
    )
    
    plt.title('Combined Weekly Load vs Stress Level by Gender')
    plt.xlabel('Combined Weekly Load (Study + Project + Exercise Hours)')
    plt.ylabel('Self-Reported Stress Level')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot and the new dataset
    plt.savefig('scatter_load_vs_stress.png')
    df.to_csv('student_combined_data_load.csv', index=False)
    
    plt.show() 
    return df