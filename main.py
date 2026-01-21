import clean_student_data as csd
import difference_calculation as dc
import genders as g
import t_test_validity as ttv
import t_test as tt
import pandas as pd

if __name__ == "__main__":
    FILE_NAME = 'student_health_data.csv'
    STRESS_BIO_COL = 'Stress_Level_Biosensor'
    SELF_REPORT_COL = 'Stress_Level_Self_Report'    
    STRESS_DIFF_COL = 'Stress_Difference'
    GENDER_COL = 'Gender'
    df_cleaned = csd.process_and_clean_data(FILE_NAME)
    df_cleaned = g.standardize_gender_column(df_cleaned)
    csd.visualize_data_distribution(df_cleaned)
    df_cleaned = dc.add_difference_column(df_cleaned, SELF_REPORT_COL, STRESS_BIO_COL, STRESS_DIFF_COL)
    male_series, female_series = g.get_gender_series(df_cleaned, STRESS_DIFF_COL, GENDER_COL)
    if (ttv.verify_ttest_assumptions(male_series, female_series) is False):
        print("T-test assumptions not met. exiting.")
        exit(1)
    t_stat, p_val, effect_size = tt.independent_t_test(male_series, female_series)
    print(f"P-value: {p_val}")
    if (p_val > 0.05):
        print('resultes insgnificant')
    else:
        print('resultes are sgnificant, effect size is:', {effect_size})
        tt.conclusion_effect(effect_size)
    