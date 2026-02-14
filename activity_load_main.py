
import logging
import clean_student_data as csd
import difference_calculation as dc
import genders as g
import activity_load_process as alp
import clean_student_data as csd
import genders as g
import interaction_regression as ir
import load_outliers_detection_and_visualization as lov

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    FILE_NAME = 'student_health_data.csv'
    COMBINED_LOAD_COL = 'Combined_Weekly_Load'
    GENDER_COL = 'Gender'
    MALE_LABEL = 'M'
    FEMALE_LABEL = 'F'
    
    SELF_REPORT_COL = 'Stress_Level_Self_Report'    
    STRESS_BIO_COL = 'Stress_Level_Biosensor'

    gender_cols = ['gender', 'sex', 'user_sex']
    gender_rules = {
    MALE_LABEL: ['m', 'male', 'man', '1'],
    FEMALE_LABEL: ['f', 'female', 'woman', '0']
    }

    df = csd.process_and_clean_data(FILE_NAME)
    df = g.standardize_categorical_column(
    df, 
    gender_cols, 
    gender_rules, 
    GENDER_COL
)

    out, df_cleaned = csd.complete_df_without_outliers(df)

    print(f"Removed {len(out)} outliers based on IQR method.")
    print(f"{(out)}")
    condition_mask = df_cleaned['Blood_Pressure_Systolic'] < df_cleaned['Blood_Pressure_Diastolic']
    df_removed = df_cleaned[condition_mask]
    df_cleaned = df_cleaned[~condition_mask]


    print(f"Removed {len(df_removed)} outliers based on IQR method.")
    print(f"{(df_removed)}")

    print(f"{len(df_cleaned)}")
    df_cleaned = alp.process_and_visualize_load(df_cleaned)
    df_no_outliers = lov.Total_daily_load_outliers(df_cleaned, GENDER_COL, COMBINED_LOAD_COL)
    
    print(f"{len(df_cleaned)}")
    lov.total_daily_load_boxplot(df_no_outliers, GENDER_COL, COMBINED_LOAD_COL, MALE_LABEL, FEMALE_LABEL)
    lov.status_image(df_no_outliers, GENDER_COL, MALE_LABEL, FEMALE_LABEL, STRESS_BIO_COL, COMBINED_LOAD_COL)
    model = ir.run_interaction_regression(
        df_no_outliers,
        COMBINED_LOAD_COL,
        STRESS_BIO_COL,
        GENDER_COL
    )
    ir.plot_interaction(df_no_outliers, COMBINED_LOAD_COL, STRESS_BIO_COL, GENDER_COL)

    ir.analyze_simple_slopes(df_no_outliers, COMBINED_LOAD_COL, STRESS_BIO_COL, GENDER_COL)

    ir.check_regression_assumptions(model)

