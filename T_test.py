import pandas as  pd
import numpy as np
from scipy import stats
import pingouin as pg

def remove_col_outliers(df,df_column): 
    Q1 = df[df_column].quantile(0.25) #Lower quartile
    Q3 = df[df_column].quantile(0.75) #Upper quartile
    IQR = Q3 - Q1

    lower_lim = Q1 - 1.5 * IQR
    upper_lim = Q3 + 1.5 * IQR

    #Using mask to filter out rows containing outliers
    #Outlier rows are defined as values outside the IQR-based limits
    df_no_outliers = df[
        (df[df_column] >= lower_lim) &
        (df[df_column] <= upper_lim)]
    
    return df_no_outliers




def independent_t_test(male_diff_series,female_diff_series):
    t_stat, p_val = stats.ttest_ind(male_diff_series, female_diff_series)
    if (p_val > 0.05):
        print("Fail to reject null hypothesis: No significant difference between groups.")
        return t_stat, p_val, None
    effect_size=pg.compute_effsize(male_diff_series, female_diff_series)
    return t_stat, p_val, effect_size



def conclusion_effect(effect_size):
    effect_size_abs=abs(effect_size)
    if effect_size_abs<0.2:
        print("No effect")
    elif 0.2<=effect_size_abs<0.5:
        print("Weak effect of "+str(effect_size)) 
    elif 0.5<=effect_size_abs<0.8:
        print("Medium effect of "+str(effect_size))   
    else:
        print("Strong effect of "+str(effect_size))        






        

    




    





    

