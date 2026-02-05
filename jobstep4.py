#%%
# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling

#%%
# Function to change columns to categorical
# df: DataFrame to be changed
# datatype: string name of datatype to change to, in this case 'category'
def category_change(df,datatype:str):
    category_list = list(df.select_dtypes(datatype))
    df[category_list] = df[category_list].astype('category')
    return df
#%%
# Function to normalize the numerical data
# df: DataFrame to be changed
# datatype: string name of datatype to change to, in this case 'number'
def normalize(df,datatype:str):
    numeric = list(df.select_dtypes(datatype))
    df[numeric] = MinMaxScaler().fit_transform(df[numeric])
    return df
#%%
# Function to one-hot encode factor variables to binary columns
def one_hot_encoding(df,datatype:str):
    category_list = list(df.select_dtypes(datatype))
    df_encoded = pd.get_dummies(df,columns=category_list)
    return df_encoded
#%%
# Function to drop columns that are not necessary to computation
def drop(df,cols_to_drop:list):
    df_clean = df.drop(cols_to_drop,axis=1)
    return df_clean
#%%
# Function to separate target variable into two bins to create a binary target variable column
# target: name of the column of the target variable
# new_col: name of the new column that the binary target variable will be in
# percentile: float value that is the cutoff point for the two bins, usually 75th percentile
def two_bins(df,target:str,new_col:str,percentile:float):
    df[new_col] = pd.cut(df[target],
                         bins=[-1,percentile,1],
                         labels=[0,1])
    return df
#%%
# Function to calculate and return prevalence of target variable
# binary_target: string name of column with binary target variable
def prevalence(df,binary_target:str):
    prev = (df[binary_target].value_counts()[1]/len(df[binary_target]))
    return prev
#%%
# Function to split binary target variable data into training, testing, and tuning datasets
# binary_target: string name of column with binary target variable
# first_split: integer of how many values to put into training (usually # of 0's)
# second_split: float for proportion of values to go into test and tune (usually .5)
def split_data(df,binary_target:str,first_split:int,second_split:float):
    df = df.dropna(subset=[binary_target])
    train,test = train_test_split(
        df,
        train_size=first_split,
        stratify=df[binary_target]
    )
    tune,test = train_test_split(
        test,
        train_size=.5,
        stratify=test[binary_target]
    )
    return train,test,tune
#%%
# TESTING DOWN HERE.
jobdf = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
# %%
df = category_change(jobdf,'string')
# %%
df = normalize(df,'number')
# %%
df = one_hot_encoding(df,'category')
# %%
df = drop(df,['sl_no','ssc_b_Central','ssc_b_Others','hsc_b_Central','hsc_b_Others'])
# %%
df = two_bins(df,'mba_p','mba_p_f',0.5639)
# %%
prev = prevalence(df,'mba_p_f')
print(f"Prevalence: {prev}")
# %%
train,test,tune = split_data(df,'mba_p_f',161,.5)
# %%
print(f"Training set shape: {train.shape}")
print(f"Testing set shape: {test.shape}")
print(f"Tuning set shape: {tune.shape}")
df
# %%
# All steps complete!