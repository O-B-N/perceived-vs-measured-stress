import logging
import clean_student_data as csd
import difference_calculation as dc
import genders as g
import t_test_validity as ttv
import t_test as tt
import pandas as pd

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    FILE_NAME = 'student_health_data.csv'
    STRESS_BIO_COL = 'Stress_Level_Biosensor'
    SELF_REPORT_COL = 'Stress_Level_Self_Report'    
    STRESS_DIFF_COL = 'Stress_Difference'
    GENDER_COL = 'Gender'
    MALE_LABEL = 'M'
    FEMALE_LABEL = 'F'

    gender_cols = ['gender', 'sex', 'user_sex']
    gender_rules = {
    MALE_LABEL: ['m', 'male', 'man', '1'],
    FEMALE_LABEL: ['f', 'female', 'woman', '0']
    }

    df_cleaned = csd.process_and_clean_data(FILE_NAME)
    df_cleaned = g.standardize_categorical_column(
    df_cleaned, 
    gender_cols, 
    gender_rules, 
    GENDER_COL
)
    
    
    out, df_cleaned = csd.complete_df_without_outliers(df_cleaned)

    print(f"Removed {len(out)} outliers based on IQR method.")
    print(f"{(out)}")
    condition_mask = df_cleaned['Blood_Pressure_Systolic'] < df_cleaned['Blood_Pressure_Diastolic']
    df_removed = df_cleaned[condition_mask]
    df_cleaned = df_cleaned[~condition_mask]


    print(f"Removed {len(df_removed)} outliers based on IQR method.")
    print(f"{(df_removed)}")

    print(f"{len(df_cleaned)}")



    csd.visualize_data_distribution(df_cleaned)

    df_cleaned = dc.add_difference_column(df_cleaned, SELF_REPORT_COL, STRESS_BIO_COL, STRESS_DIFF_COL)
    
    df_cleaned = tt.outliers_IQR(df_cleaned, STRESS_DIFF_COL)

    print(f"{len(df_cleaned)}")
    
    male_series, female_series = g.get_gender_series(df_cleaned, STRESS_DIFF_COL, GENDER_COL) #לקרוא לזה מהפונקציות 
    if (ttv.verify_ttest_assumptions(male_series, female_series) is False):
        logger.error(f"T-test assumptions not met. exiting.")
        exit(1)

    t_stat, p_val = tt.independent_t_test(df_cleaned, STRESS_DIFF_COL, GENDER_COL, MALE_LABEL, FEMALE_LABEL)
    logger.info(f"P value: {p_val}")
    tt.conclusion_ttest_ind(t_stat,p_val)
    tt.conclusion_effect(tt.effect_size(df_cleaned, STRESS_DIFF_COL, GENDER_COL, MALE_LABEL, FEMALE_LABEL))
    