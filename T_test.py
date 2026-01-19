import pandas as  pd
import numpy as np
from scipy import stats
import pingouin as pg

def outliers_diff_report_biosensor(df_with_diff_column): #The input will be avigail fonction output
    Q1 = df_with_diff_column["diff_column"].quantile(0.25) #Lower quartile
    Q3 = df_with_diff_column["diff_column"].quantile(0.75) #Upper quartile
    IQR = Q3 - Q1

    lower_lim = Q1 - 1.5 * IQR
    upper_lim = Q3 + 1.5 * IQR

    #Using mask to filter out rows containing outliers
    #Outlier rows are defined as values outside the IQR-based limits
    df_no_outliers = df_with_diff_column[
        (df_with_diff_column["diff_column"] >= lower_lim) &
        (df_with_diff_column["diff_column"] <= upper_lim)]
    
    return df_no_outliers



def independent_t_test(df, diff_col="Diff", gender_col="Gender", male_label="M",female_label="F"):
    male_diff_series = df.loc[df[gender_col] == male_label, diff_col] 
    female_diff_series = df.loc[df[gender_col] == female_label, diff_col]
    t_stat, p_val = stats.ttest_ind(male_diff_series, female_diff_series)

    return t_stat, p_val



def conclusion_ttest_ind(t_stat,p_val): #two- tailed test!!!
    if p_val<0.05:
        print("The results were statistically significant at the 0.05 level.")
    else:
        print("The results were not statistically significant at the 0.05 level.")
        if p_val<0.1:
            print("The results were statistically significant at the 0.1 level.")
        else:
            print("The results were not statistically significant at the 0.1 level.")    


def effect_size(df, diff_col="Diff", gender_col="Gender", male_label="M",female_label="F"):
    male_diff_series= df.loc[df[gender_col] == male_label, diff_col] 
    female_diff_series= df.loc[df[gender_col] == female_label, diff_col]
    effect_size=pg.compute_effsize(male_diff_series, female_diff_series)
    return effect_size



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






        

    




    





    

