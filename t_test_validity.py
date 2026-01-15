from scipy import stats
import logging

def verify_ttest_assumptions(group1_data: pd.Series, group2_data: pd.Series):
    """
    Checks the assumptions for an independent t-test:
    1. Sample size (n >= 30) OR Normality (Shapiro-Wilk test).
    2. Equality of variances (Levene's test).
    """
    logger = logging.getLogger(__name__)
    results = {"normality": False, "equal_variance": False}

    # 1. Check Sample Size or Normality
    n1, n2 = len(group1_data), len(group2_data)
    
    if n1 >= 30 and n2 >= 30:
        logger.info(f"Sample sizes are sufficient (n1={n1}, n2={n2}). Central Limit Theorem applies.")
        results["normality"] = True
    else:
        # Perform Shapiro-Wilk test for normality (p > 0.05 means normal)
        _, p_norm1 = stats.shapiro(group1_data)
        _, p_norm2 = stats.shapiro(group2_data)
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            logger.info("Data follows a normal distribution (Shapiro-Wilk p > 0.05).")
            results["normality"] = True
        else:
            logger.warning("Normality assumption violated and sample size is small.")

    # 2. Check Equality of Variances (Levene's Test)
    # H0: Variances are equal. If p > 0.05, we do not reject H0.
    _, p_levene = stats.levene(group1_data, group2_data)
    if p_levene > 0.05:
        logger.info(f"Equal variance assumption met (Levene's p = {p_levene:.4f}).")
        results["equal_variance"] = True
    else:
        logger.warning(f"Variances are not equal (Levene's p = {p_levene:.4f}). Consider Welch's t-test.")

    return results