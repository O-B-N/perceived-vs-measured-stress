import statsmodels.formula.api as ols
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def run_interaction_regression(df, x_col, y_col, group_col):
    """
    Runs a multiple linear regression with interaction.
    Formula: y ~ x * group
    """
    formula = f"{y_col} ~ {x_col} * C({group_col})"
    model = ols.ols(formula, data=df).fit()
    logger.info(model.summary())
    return model

def plot_interaction(df, x_col, y_col, group_col):

    sns.lmplot(x=x_col, y=y_col, hue=group_col, data=df, height=6, aspect=1.5)
    plt.subplots_adjust(top=0.9)
    
    plt.title(f'Interaction Effect: Does the gender affects change how combined weekly load affects biosensor stress?')
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