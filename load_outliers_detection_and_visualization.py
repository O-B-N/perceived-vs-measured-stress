import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def total_daily_load_boxplot(df,gender_col,load_col,male_label,female_label):

    #Varify that the columns exist
    for col in [gender_col, load_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")  #To see which column doesnt exist

    sub_df_gender_load= df[[gender_col, load_col]].copy() #Create a copy of two column
    sub_df_gender_load[load_col]= pd.to_numeric(sub_df_gender_load[load_col], errors="coerce")  #Convert the load column to numeric values, coercing invalid entries to NaN
    clean_sub_df_gender_load = sub_df_gender_load.dropna(subset=[load_col]) #To clean NaN from load column
    series_women_men_count = clean_sub_df_gender_load.groupby(gender_col)[load_col].count() #Group the data by gender and count non-missing load values in each gender


    if series_women_men_count.min() == 0: #If there are no numerical load values for men/women - stop the code .
        raise ValueError("There are not enough numerical values to draw a boxplot.")

    if series_women_men_count.min() < 20: #If there are least than 20 numerical load values for men/women - send a message and draw
        print("Pay attention: Not enough numeric values are available for at least one gender.")


    sns.boxplot(x=gender_col,y=load_col,data=clean_sub_df_gender_load) #Plot box for each gender
    plt.xlabel(gender_col)
    plt.ylabel("Combined Daily load (hours)")
    plt.title("Combined Daily load by Gender")
    plt.show()

def Total_daily_load_outliers(df,gender_col,load_col):
    for col in [gender_col, load_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.") #To see which column doesnt exist
    df_copy=df.copy()
    list_df=[]
    for gender, df_for_gender in df_copy.groupby(gender_col): #Loop over each gender group from df
        list_df.append(outliers_IQR(df_for_gender,load_col)) # Remove outliers from the current gender group using the IQR method
    return pd.concat(list_df, axis=0)



def status_image(df_without_outliers,gender_col, male_label,female_label,stress_col,load_col):
    females_rows_df = df_without_outliers[df_without_outliers[gender_col] == female_label]  # Df with Females rows
    males_rows_df = df_without_outliers[df_without_outliers[gender_col] == male_label] #Df with Males rows
    females_and_males_two_df= [females_rows_df,males_rows_df] #List with df of Females and df of Males
    females_and_males_title=["Females","Males"]
    for df_gender,title_gender in zip(females_and_males_two_df,females_and_males_title): # Loop over each gender DataFrame with its corresponding label
        plt.figure(figsize=(20, 7))
        plt.scatter(df_gender[load_col],df_gender[stress_col],s=12)
        plt.xlabel("Load")
        plt.ylabel("Sensor stress")
        plt.title("Stress-Load - "+title_gender)
        plt.show()

