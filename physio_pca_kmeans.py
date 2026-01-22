import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

import logging


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# 1) Load data
def load_data():
    logger.info("Loading dataset: cleaned_student_health_data.csv")
    df = pd.read_csv("cleaned_student_health_data.csv")
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    return df


# 2) Select physiological features and handle missing values
def select_physiological_features(df):
    physio_features = [
        "Heart_Rate",
        "Blood_Pressure_Systolic",
        "Blood_Pressure_Diastolic",
        "Stress_Level_Biosensor"
    ]

    logger.info(f"Selected physiological features: {physio_features}")
    X = df[physio_features].copy()

    logger.info("Checking missing values in selected features:")
    logger.info(f"\n{X.isnull().sum()}")

    if X.isnull().any().any():
        logger.warning("Missing values found in physiological features. Filling with median.")
        X = X.fillna(X.median(numeric_only=True))

    return X, physio_features


# 3) Standardize features
def standardize_features(df, X, physio_features):
    logger.info("Standardizing physiological features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=physio_features, index=df.index)

    logger.info("Standardization check (means should be ~0):")
    logger.info(f"\n{X_scaled_df.mean()}")

    return X_scaled_df


# 4) Run PCA and log results
def run_pca(X_scaled_df, physio_features):
    logger.info("Running PCA on standardized physiological features...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled_df)

    explained_variance = pca.explained_variance_ratio_
    logger.info("Explained variance ratio per component:")
    logger.info(explained_variance)
    logger.info(f"Sum of explained variance: {explained_variance.sum()}")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=physio_features,
        columns=[f"PC{i+1}" for i in range(len(physio_features))]
    )
    logger.info("PCA loadings:")
    logger.info(f"\n{loadings}")

    return pca, X_pca, explained_variance, loadings


# Plot PCA explained variance
def plot_pca_variance(explained_variance):
    logger.info("Plotting cumulative explained variance...")
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(explained_variance) + 1),
        np.cumsum(explained_variance),
        marker='o'
    )
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.tight_layout()
    plt.show()

    logger.info("Plotting explained variance by component...")
    plt.figure(figsize=(6, 4))
    plt.bar(
        range(1, len(explained_variance) + 1),
        explained_variance
    )
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.ylim(0, 1)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance by PCA Components")
    plt.tight_layout()
    plt.show()


# 5) KMeans - Elbow method
def elbow_method_kmeans(X_scaled_df):
    logger.info("Evaluating number of clusters using Elbow method...")
    inertia = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans_tmp = KMeans(n_clusters=k, random_state=0)
        kmeans_tmp.fit(X_scaled_df)
        inertia.append(kmeans_tmp.inertia_)

    logger.info(f"Inertia values (k=1..10): {inertia}")

    logger.info("Plotting Elbow curve (inertia vs k)...")
    plt.figure(figsize=(6, 4))
    plt.plot(list(K_range), inertia, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for KMeans")
    plt.tight_layout()
    plt.show()

    return inertia, K_range


# 6) Final KMeans clustering
def run_final_kmeans(df, X_scaled_df, physio_features, k_final=3):
    logger.info(f"Running final KMeans with k={k_final}...")
    kmeans = KMeans(n_clusters=k_final, random_state=0)
    df["Cluster"] = kmeans.fit_predict(X_scaled_df)

    logger.info("Cluster sizes:")
    logger.info(f"\n{df['Cluster'].value_counts().sort_index()}")

    cluster_means = df.groupby("Cluster")[physio_features].mean()
    logger.info("Mean physiological values per cluster (original units):")
    logger.info(f"\n{cluster_means}")

    return df, kmeans, cluster_means


# 7) Post-hoc comparisons
def post_hoc_comparisons(df):
    logger.info("Mean self-reported stress by cluster:")
    logger.info(df.groupby("Cluster")["Stress_Level_Self_Report"].mean())

    logger.info("Gender distribution by cluster (row-normalized):")
    logger.info(f"\n{pd.crosstab(df['Cluster'], df['Gender'], normalize='index')}")

    logger.info("Sleep quality distribution by cluster (row-normalized):")
    logger.info(f"\n{pd.crosstab(df['Cluster'], df['Sleep_Quality'], normalize='index')}")


# 8) Plot post-hoc comparisons
def plot_post_hoc(df):
    logger.info("Plotting self-reported stress by cluster (bar plot)...")
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="Cluster",
        y="Stress_Level_Self_Report",
        ci=None
    )
    plt.xlabel("Cluster")
    plt.ylabel("Self-Reported Stress")
    plt.title("Self-Reported Stress by Physiological Cluster")
    plt.tight_layout()
    plt.show()

    logger.info("Plotting gender distribution by cluster (stacked bar)...")
    gender_ct = pd.crosstab(df["Cluster"], df["Gender"], normalize="index")
    gender_ct.plot(kind="bar", stacked=True)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.title("Gender Distribution by Physiological Cluster")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()

    logger.info("Plotting sleep quality distribution by cluster (stacked bar)...")
    sleep_ct = pd.crosstab(df["Cluster"], df["Sleep_Quality"], normalize="index")

    desired_order = ["Good", "Moderate", "Poor"]
    existing_order = [c for c in desired_order if c in sleep_ct.columns]
    sleep_ct = sleep_ct[existing_order]

    sleep_ct.plot(kind="bar", stacked=True)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.title("Sleep Quality Distribution by Physiological Cluster")
    plt.legend(title="Sleep Quality")
    plt.tight_layout()
    plt.show()

#########################################################################################
def main():
    df = load_data()
    X, physio_features = select_physiological_features(df)
    X_scaled_df = standardize_features(df, X, physio_features)

    pca, X_pca, explained_variance, loadings = run_pca(X_scaled_df, physio_features)
    plot_pca_variance(explained_variance)

    inertia, K_range = elbow_method_kmeans(X_scaled_df)

    df, kmeans, cluster_means = run_final_kmeans(df, X_scaled_df, physio_features, k_final=3)

    post_hoc_comparisons(df)
    plot_post_hoc(df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
