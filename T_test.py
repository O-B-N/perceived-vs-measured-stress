import pandas as  pd
import numpy as np
from scipy import stats
import pingouin as pg
import genders as g
import logging

logger= logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def outliers_IQR(df,df_column): 
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



def independent_t_test(df, diff_col, gender_col, male_label,female_label):
    male_diff_series = df.loc[df[gender_col] == male_label, diff_col] 
    female_diff_series = df.loc[df[gender_col] == female_label, diff_col]
    t_stat, p_val = stats.ttest_ind(male_diff_series, female_diff_series)

    return t_stat, p_val



def conclusion_ttest_ind(t_stat,p_val): #two- tailed test!!!
    if p_val<0.05:
        logger.info("The results were statistically significant at the 0.05 level.")
    else:
        logger.info("The results were not statistically significant at the 0.05 level.")
        if p_val<0.1:
            logger.info("The results were statistically significant at the 0.1 level.")
        else:
            logger.info("The results were not statistically significant at the 0.1 level.")    


def effect_size(df, diff_col, gender_col, male_label,female_label):
    male_diff_series= df.loc[df[gender_col] == male_label, diff_col] 
    female_diff_series= df.loc[df[gender_col] == female_label, diff_col]
    effect_size=pg.compute_effsize(male_diff_series, female_diff_series)
    return effect_size



def conclusion_effect(effect_size):
    effect_size_abs=abs(effect_size)
    if effect_size_abs<0.2:
        logger.info("No effect")
    elif 0.2<=effect_size_abs<0.5:
        logger.info("Weak effect of "+str(effect_size)) 
    elif 0.5<=effect_size_abs<0.8:
        logger.info("Medium effect of "+str(effect_size))   
    else:
        logger.info("Strong effect of "+str(effect_size))