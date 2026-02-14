import sys
import pandas as pd
import logging 
import physio_pca_kmeans as ppk
import clean_student_data as csd
import difference_calculation as dc
import genders as g
import activity_load_process as alp
import interaction_regression as ir
import load_outliers_detection_and_visualization as lov
import t_test_validity as ttv
import t_test as tt

# Constants
FILE_NAME = 'student_health_data.csv'
COMBINED_LOAD_COL = 'Combined_Weekly_Load'
STRESS_BIO_COL = 'Stress_Level_Biosensor'
SELF_REPORT_COL = 'Stress_Level_Self_Report'    
STRESS_DIFF_COL = 'Stress_Difference'
GENDER_COL = 'Gender'
MALE_LABEL = 'M'
FEMALE_LABEL = 'F'
# Gender standardization configs
gender_cols = ['gender', 'sex', 'user_sex']
gender_rules = {
MALE_LABEL: ['m', 'male', 'man', '1'],
FEMALE_LABEL: ['f', 'female', 'woman', '0']
}

# Logging setup
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("project_run.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

import sys
import logging
import pandas as pd

# Import your modules
import clean_student_data as csd
import difference_calculation as dc
import genders as g
import activity_load_process as alp
import interaction_regression as ir
import load_outliers_detection_and_visualization as lov
import t_test_validity as ttv
import t_test as tt
import physio_pca_kmeans as ppk

# --- Constants ---
FILE_NAME = 'student_health_data.csv'
COMBINED_LOAD_COL = 'Combined_Weekly_Load'
STRESS_BIO_COL = 'Stress_Level_Biosensor'
SELF_REPORT_COL = 'Stress_Level_Self_Report'    
STRESS_DIFF_COL = 'Stress_Difference'
GENDER_COL = 'Gender'
MALE_LABEL = 'M'
FEMALE_LABEL = 'F'

# Gender standardization configs
GENDER_COLS = ['gender', 'sex', 'user_sex']
GENDER_RULES = {
    MALE_LABEL: ['m', 'male', 'man', '1'],
    FEMALE_LABEL: ['f', 'female', 'woman', '0']
}

def setup_logging():
    """Configures the root logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("project_run.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_stress_difference_analysis(df, logger):
    logger.info("--- Part 1: Stress Difference Analysis ---")
    df_diff = dc.add_difference_column(df.copy(), SELF_REPORT_COL, STRESS_BIO_COL, STRESS_DIFF_COL)
    df_diff = tt.outliers_IQR(df_diff, STRESS_DIFF_COL)
    
    male_series, female_series = g.get_gender_series(df_diff, STRESS_DIFF_COL, GENDER_COL)
    
    if not ttv.verify_ttest_assumptions(male_series, female_series):
        logger.warning("T-test assumptions not met (proceeding with caution).")
    
    t_stat, p_val = tt.independent_t_test(df_diff, STRESS_DIFF_COL, GENDER_COL, MALE_LABEL, FEMALE_LABEL)
    logger.info(f"T-Test Results - P value: {p_val}")
    tt.conclusion_ttest_ind(t_stat, p_val)
    effect = tt.effect_size(df_diff, STRESS_DIFF_COL, GENDER_COL, MALE_LABEL, FEMALE_LABEL)
    tt.conclusion_effect(effect)

def run_interaction_regression_analysis(df, logger):
    logger.info("--- Part 2: Interaction Regression Analysis ---")
    df_load = alp.process_and_visualize_load(df.copy())
    df_load = lov.Total_daily_load_outliers(df_load, GENDER_COL, COMBINED_LOAD_COL)
    df_load = tt.outliers_IQR(df_load, COMBINED_LOAD_COL)

    lov.total_daily_load_boxplot(df_load, GENDER_COL, COMBINED_LOAD_COL, MALE_LABEL, FEMALE_LABEL)
    lov.plot_stress_vs_load(df_load, GENDER_COL, MALE_LABEL, FEMALE_LABEL, STRESS_BIO_COL, COMBINED_LOAD_COL)
    
    model = ir.run_interaction_regression(df_load, COMBINED_LOAD_COL, STRESS_BIO_COL, GENDER_COL)
    ir.plot_interaction(df_load, COMBINED_LOAD_COL, STRESS_BIO_COL, GENDER_COL)
    ir.analyze_simple_slopes(df_load, COMBINED_LOAD_COL, STRESS_BIO_COL, GENDER_COL)

def run_physio_analysis(df, logger):
    logger.info("--- Part 3: Physiological Features Analysis ---")
    X, physio_features = ppk.select_physiological_features(df)
    X_scaled_df = ppk.standardize_features(df, X, physio_features)

    _, _, explained_variance, _ = ppk.run_pca(X_scaled_df, physio_features)
    ppk.plot_pca_variance(explained_variance)
    ppk.elbow_method_kmeans(X_scaled_df)
    
    df_res, _, _ = ppk.run_final_kmeans(df, X_scaled_df, physio_features, k_final=3)
    ppk.post_hoc_comparisons(df_res)
    ppk.plot_post_hoc(df_res)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Data Pipeline...")
        df = csd.process_and_clean_data(FILE_NAME)
        df = g.standardize_categorical_column(df, GENDER_COLS, GENDER_RULES, GENDER_COL)
        _, df = csd.complete_df_without_outliers(df)
        csd.visualize_data_distribution(df)

        run_stress_difference_analysis(df, logger)
        run_interaction_regression_analysis(df, logger)
        run_physio_analysis(df, logger)
        
        logger.info("Analysis Pipeline Completed Successfully.")
        
    except FileNotFoundError:
        logger.error(f"File {FILE_NAME} not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()