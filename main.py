import sys
import physio_pca_kmeans as ppk
import logging
import clean_student_data as csd
import difference_calculation as dc
import genders as g
import activity_load_process as alp
import interaction_regression as ir
import load_outliers_detection_and_visualization as lov
import t_test_validity as ttv
import t_test as tt


logger = logging.getLogger(__name__)

FILE_NAME = 'student_health_data.csv'
COMBINED_LOAD_COL = 'Combined_Weekly_Load'
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

# Logging setup
def setup_logging():
    """
    Configures the root logger for the entire project.
    Logs will be sent to both the console (stdout) and a file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("project_run.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )


if __name__ == "__main__":

    # Initialize logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Data loading, and basic cleaning
    try:
        df_cleaned = csd.process_and_clean_data(FILE_NAME)
        print("Data processed and cleaned successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{FILE_NAME}' was not found. Exiting program.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}. Exiting program.")
        sys.exit(1)
    
    # Standardize the gender column
    df_cleaned = g.standardize_categorical_column(
    df_cleaned, 
    gender_cols, 
    gender_rules, 
    GENDER_COL
)
    # Remove outliers and visualize data distribution
    out, df_cleaned = csd.complete_df_without_outliers(df_cleaned)
    csd.visualize_data_distribution(df_cleaned)

    
    logger.info(f"################ part 1: Stress Difference Analysis ################")

    # add stress difference column to the df and remove outliers from it
    df_with_stress_difference = dc.add_difference_column(df_cleaned.copy(), SELF_REPORT_COL, STRESS_BIO_COL, STRESS_DIFF_COL)
    df_with_stress_difference = tt.outliers_IQR(df_with_stress_difference, STRESS_DIFF_COL)
    
    # Perform independent t-test on the stress difference
    male_series, female_series = g.get_gender_series(df_with_stress_difference, STRESS_DIFF_COL, GENDER_COL)
    if (ttv.verify_ttest_assumptions(male_series, female_series) is False):
        logger.error(f"T-test assumptions not met.")
    else:
        t_stat, p_val = tt.independent_t_test(df_with_stress_difference, STRESS_DIFF_COL, GENDER_COL, MALE_LABEL, FEMALE_LABEL)
        logger.info(f"P value: {p_val}")
        tt.conclusion_ttest_ind(t_stat,p_val)
        tt.conclusion_effect(tt.effect_size(df_with_stress_difference, STRESS_DIFF_COL, GENDER_COL, MALE_LABEL, FEMALE_LABEL))
    
    logger.info(f"################ part 2: Interaction Regression Analysis ################")

    # add combined load column to the df and remove outliers from it
    df_with_combained_load = alp.process_and_visualize_load(df_cleaned.copy())
    df_with_combained_load = lov.Total_daily_load_outliers(df_with_combained_load, GENDER_COL, COMBINED_LOAD_COL)
    df_with_combained_load = tt.outliers_IQR(df_with_combained_load, COMBINED_LOAD_COL)

    # Visualize the relationship between combined load and biosensor stress, and run the interaction regression analysis
    lov.total_daily_load_boxplot(df_with_combained_load, GENDER_COL, COMBINED_LOAD_COL, MALE_LABEL, FEMALE_LABEL)
    lov.plot_stress_vs_load(df_with_combained_load, GENDER_COL, MALE_LABEL, FEMALE_LABEL, STRESS_BIO_COL, COMBINED_LOAD_COL)
    model = ir.run_interaction_regression(
        df_with_combained_load,
        COMBINED_LOAD_COL,
        STRESS_BIO_COL,
        GENDER_COL
    )
    ir.plot_interaction(df_with_combained_load, COMBINED_LOAD_COL, STRESS_BIO_COL, GENDER_COL)
    ir.analyze_simple_slopes(df_with_combained_load, COMBINED_LOAD_COL, STRESS_BIO_COL, GENDER_COL)
    ir.check_regression_assumptions(model)


    logger.info("################ Starting part 3: Physiological Features Analysis ################")

    X, physio_features = ppk.select_physiological_features(df_cleaned)
    X_scaled_df = ppk.standardize_features(df_cleaned, X, physio_features)

    pca, X_pca, explained_variance, loadings = ppk.run_pca(X_scaled_df, physio_features)
    ppk.plot_pca_variance(explained_variance)

    inertia, K_range = ppk.elbow_method_kmeans(X_scaled_df)

    df, kmeans, cluster_means = ppk.run_final_kmeans(df_cleaned, X_scaled_df, physio_features, k_final=3)

    ppk.post_hoc_comparisons(df)
    ppk.plot_post_hoc(df)

    logger.info("Done.")
