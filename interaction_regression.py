import statsmodels.formula.api as ols
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import pandas as pd

def run_interaction_regression(df, x_col, y_col, group_col):
    """
    Runs a multiple linear regression with interaction.
    Formula: y ~ x * group
    """
    # C() tells statsmodels that 'group_col' is Categorical (e.g., Male/Female)
    formula = f"{y_col} ~ {x_col} * C({group_col})"
    
    model = ols.ols(formula, data=df).fit()
    
    print(model.summary())

    return model

def plot_interaction(df, x_col, y_col, group_col):

    sns.lmplot(x=x_col, y=y_col, hue=group_col, data=df, height=6, aspect=1.5)
    plt.subplots_adjust(top=0.9)
    
    plt.title(f'Interaction Effect: Does the gender affects change how combined weekly load affects biosensor stress?')
    plt.show()

def check_regression_assumptions(model):
    """
    Checks the assumptions of OLS linear regression:
    1. Linearity
    2. Homoscedasticity (Constant Variance)
    3. Normality of Residuals
    4. Independence of Errors
    
    Args:
        model: A fitted statsmodels model object.
    """
    print("\n--- Regression Assumptions Check ---")
    
    # Extract residuals and fitted values
    residuals = model.resid
    fitted_vals = model.fittedvalues
    
    # --- 1. Statistical Tests ---
    
    # A. Normality Test (Shapiro-Wilk)
    # H0: Data is normally distributed
    shapiro_test = stats.shapiro(residuals)
    print(f"1. Normality (Shapiro-Wilk): W={shapiro_test[0]:.4f}, p={shapiro_test[1]:.4f}")
    if shapiro_test[1] < 0.05:
        print("   Warning: Residuals are NOT normally distributed (p < 0.05).")
    else:
        print("   Pass: Residuals look normal.")

    # B. Independence Test (Durbin-Watson)
    # Values range from 0 to 4. 2 is no autocorrelation. <1.5 or >2.5 is cause for concern.
    dw_val = sm.stats.stattools.durbin_watson(residuals)
    print(f"2. Independence (Durbin-Watson): {dw_val:.4f}")
    if dw_val < 1.5 or dw_val > 2.5:
        print("   Warning: Potential Autocorrelation detected.")
    else:
        print("   Pass: No strong autocorrelation detected.")

    # --- 2. Diagnostic Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diagnostic Plots for Regression Assumptions', fontsize=16)

    # Plot 1: Residuals vs Fitted (Check for Linearity & Homoscedasticity)
    # We want to see a random cloud of points around 0, no patterns (like a funnel).
    sns.scatterplot(x=fitted_vals, y=residuals, ax=axes[0, 0], alpha=0.6)
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')

    # Plot 2: Histogram of Residuals (Check for Normality)
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color='blue')
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].set_xlabel('Residuals')

    # Plot 3: Q-Q Plot (Check for Normality)
    # Points should fall on the red line.
    sm.qqplot(residuals, line='45', fit=True, ax=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')

    # Plot 4: Residuals vs Order (Check for Independence/Drift)
    # Relevant if data has a time component or specific order
    axes[1, 1].plot(residuals.values, marker='o', linestyle='', alpha=0.5)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Residuals vs Order (Index)')
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel('Residuals')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    


def analyze_simple_slopes(df, x_col, y_col, group_col):
    """
    Performs 'Simple Slopes Analysis' to decompose a significant interaction.
    Runs separate linear regressions for each group in 'group_col'.
    
    Args:
        df: The dataframe.
        x_col: The independent variable (Predictor/Load).
        y_col: The dependent variable (Outcome/Stress).
        group_col: The categorical moderator (Gender).
        
    Returns:
        pd.DataFrame: A summary table of slopes per group.
    """
    print(f"\n--- Post-Hoc Analysis: Simple Slopes ({y_col} ~ {x_col}) by {group_col} ---")
    
    groups = df[group_col].dropna().unique()
    results = []
    
    for group in groups:
        # Filter data for the specific group
        subset = df[df[group_col] == group]
        
        # Run regression for this specific group
        formula = f"{y_col} ~ {x_col}"
        model = ols.ols(formula, data=subset).fit()
        
        # Extract key metrics
        slope = model.params[x_col]
        p_value = model.pvalues[x_col]
        conf_int = model.conf_int().loc[x_col].values
        r_squared = model.rsquared
        n_obs = int(model.nobs)
        
        results.append({
            "Group": group,
            "N": n_obs,
            "Slope (Beta)": slope,
            "P-value": p_value,
            "Significant?": "YES" if p_value < 0.05 else "NO",
            "R-squared": r_squared,
            "CI Lower": conf_int[0],
            "CI Upper": conf_int[1]
        })
    
    # Create a clean summary table
    results_df = pd.DataFrame(results).set_index("Group")
    print(results_df)
    
    return results_df