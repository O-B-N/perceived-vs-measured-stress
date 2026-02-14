import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import physio_pca_kmeans as m


def make_test_df():
    # 12 rows so KMeans can run for k=1..10
    return pd.DataFrame({
        "Heart_Rate": [70, 80, np.nan, 90, 85, 76, 88, 92, 67, 73, 81, 95],
        "Blood_Pressure_Systolic": [120, 130, 125, np.nan, 128, 118, 140, 135, 110, 115, 132, 145],
        "Blood_Pressure_Diastolic": [80, 85, 82, 88, np.nan, 78, 90, 87, 70, 75, 83, 92],
        "Stress_Level_Biosensor": [3.0, 4.0, 2.0, 5.0, 3.5, 2.5, 4.5, 5.5, 1.5, 2.0, 3.8, 6.0],
        "Stress_Level_Self_Report": [2.0, 5.0, 1.0, 4.0, 3.0, 2.0, 4.0, 5.0, 1.0, 2.0, 3.0, 6.0],
        "Gender": ["F", "M", "F", "M", "F", "F", "M", "M", "F", "F", "M", "M"],
        "Sleep_Quality": ["Good", "Moderate", "Poor", "Good", "Moderate", "Poor",
                          "Good", "Moderate", "Poor", "Good", "Moderate", "Poor"]
    })


def disable_plots():
    # prevent pop-up windows from blocking
    m.plt.show = lambda: None


def test_select_physiological_features(df):
    X, features = m.select_physiological_features(df)
    assert list(X.columns) == features
    assert X.shape[0] == df.shape[0]
    assert not X.isnull().any().any()  # no NaN values after filling


def test_standardize_features(df):
    X, features = m.select_physiological_features(df)
    X_scaled = m.standardize_features(df, X, features)

    # mean 0
    assert np.all(np.abs(X_scaled.mean()) < 1e-10)

    # std 1
    assert np.all(np.abs(X_scaled.std(ddof=0) - 1) < 1e-10)


def test_run_pca(df):
    X, features = m.select_physiological_features(df)
    X_scaled = m.standardize_features(df, X, features)

    pca, X_pca, explained, loadings = m.run_pca(X_scaled, features)

    assert X_pca.shape == (len(df), len(features))
    assert abs(float(np.sum(explained)) - 1.0) < 1e-12
    assert list(loadings.index) == features


def test_elbow_method_kmeans(df):
    X, features = m.select_physiological_features(df)
    X_scaled = m.standardize_features(df, X, features)

    inertia, K_range = m.elbow_method_kmeans(X_scaled)

    assert len(inertia) == 10
    assert list(K_range) == list(range(1, 11))
    assert all(inertia[i] >= inertia[i + 1] for i in range(len(inertia) - 1))


def test_run_final_kmeans(df):
    X, features = m.select_physiological_features(df)
    X_scaled = m.standardize_features(df, X, features)

    df_out, kmeans, cluster_means = m.run_final_kmeans(df.copy(), X_scaled, features, k_final=3)

    assert "Cluster" in df_out.columns
    assert df_out["Cluster"].nunique() <= 3
    assert set(cluster_means.columns) == set(features)


def test_post_hoc_comparisons(df):
    df2 = df.copy()
    df2["Cluster"] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    m.post_hoc_comparisons(df2)  # just checking that no error is raised


def run_all():
    disable_plots()
    df = make_test_df()

    tests = [
        ("select_physiological_features", test_select_physiological_features),
        ("standardize_features", test_standardize_features),
        ("run_pca", test_run_pca),
        ("elbow_method_kmeans", test_elbow_method_kmeans),
        ("run_final_kmeans", test_run_final_kmeans),
        ("post_hoc_comparisons", test_post_hoc_comparisons),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn(df)
            print(f"[PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name} (assertion failed)")
            raise
        except Exception as e:
            print(f"[FAIL] {name} ({type(e).__name__}: {e})")
            raise

    print(f"\nDONE: {passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
