# Perceived vs. Measured Stress Analysis

## Project Overview

This project investigates the correlation between self-reported stress levels and sensor-measured physiological indicators (Biosensor). Utilizing a dataset of student health metrics, we analyze gender-based differences, the impact of weekly workload (study + exercise), and underlying physiological clusters using Unsupervised Learning (PCA & K-Means).

## Project Structure

The project follows a modular functional approach:

- `main.py`: The entry point orchestrating data loading, cleaning, and analysis stages.
- `clean_student_data.py` & `standardization.py`: Modules for data preprocessing, outlier removal, and categorical standardization.
- `difference_calculation.py` & `activity_load_process.py`: Feature engineering modules.
- `t_test.py`, `interaction_regression.py`, `physio_pca_kmeans.py`: Statistical and ML analysis modules.
- `load_outliers_detection_and_visualization.py`: Visualization helpers.
- `tests`: folder with unit tests
- `run_all_tests.py`: to run all tests

## Key Stages

1.  **Data Processing:** Loading CSV, standardizing 'Gender' column, and cleaning outliers using IQR.
2.  **Stress Analysis:** Independent T-tests to compare self-reported vs. biosensor stress gaps between genders.
3.  **Regression Analysis:** Interaction regression to test if Gender moderates the effect of Total Weekly Load on Stress.
4.  **Clustering:** PCA for dimensionality reduction followed by K-Means clustering on physiological features (HR, GSR, etc.).

## Installation & Usage

### Prerequisites

- Python 3.8+
- See `requirements.txt` for package list.

### Setup

1.  Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2.  Ensure `student_health_data.csv` is in the root directory.

3.  Execute the main script:

```
python main.py

```

### Tests

To run the tests:

```
python run_all_tests.py
```

### Dataset

This dataset simulates physiological, psychological, and academic data for college students, focusing on health risk assessment in high-stress environments like academic and entrepreneurial settings. It is designed to support the development and testing of machine learning models that predict occupational health risks, such as stress and physical strain. The data includes both sensor-based measurements and self-reported values.

https://www.kaggle.com/datasets/ziya07/student-health-data

### Bibliography

Calderón-García, A., Álvarez-Gallardo, E., Belinchón-deMiguel, P., & Clemente-Suárez, V. J. (2024). Gender differences in autonomic and psychological stress responses among educators: A heart rate variability and psychological assessment study. Frontiers in Psychology, 15, 1422709. https://doi.org/10.3389/fpsyg.2024.1422709

O'Connor, D. B., Thayer, J. F., & Vedhara, K. (2021). Stress and health: A review of psychobiological processes. Annual Review of Psychology, 72, 663–688. https://doi.org/10.1146/annurev-psych-062520-122331

Verma, R., Balhara, Y. P. S., & Gupta, C. S. (2011). Gender differences in stress response: Role of developmental and biological determinants. Industrial Psychiatry Journal, 20(1), 4–10. https://doi.org/10.4103/0972-6748.98407

Zintel, S., Schmidt, L. I., Neubauer, A. B., Stoffel, M., Rafiee, Y., Ditzen, B., & Sieverding, M. (2025). Daily sex differences in stress and cortisol: The role of gender-related variables. Psychoneuroendocrinology, 174, 107310. https://www.sciencedirect.com/science/article/pii/S0306453025002033

קדמון, א. (2025, 27 בינואר). לחץ דם גבוה ולחץ דם תקין. שירותי בריאות כללית. https://www.clalit.co.il/he/medical/medical_diagnosis/Pages/hypertension.aspx
