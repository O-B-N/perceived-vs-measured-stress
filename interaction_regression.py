import statsmodels.formula.api as ols

def run_interaction_regression(df, x_col, y_col, group_col):
    """
    Runs a multiple linear regression with interaction.
    Formula: y ~ x * group
    """
    # 1. Define the formula
    # C() tells statsmodels that 'group_col' is Categorical (e.g., Male/Female)
    formula = f"{y_col} ~ {x_col} * C({group_col})"
    
    # 2. Fit the model
    model = ols.ols(formula, data=df).fit()
    
    # 3. Print the full summary
    print(model.summary())
    
    return model