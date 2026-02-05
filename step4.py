#%%
# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# %%
# Cleaning the DataFrame
# df: DataFrame to be cleaned
# cat: list of column names to be categorized
# bool_col: list of column names to change columns that need to be booleans
def data_cleaning(df, cat:list, bool_col:list):
    df[bool_col] = (df[bool_col] == 'X').astype('int')
    df[cat] = df[cat].astype('category')
    return df
# %%
# Altering columns with too many groups
# col_to_remap: one column name to remap groupings
# remap: a dictionary with keys for new names and values as old names
def factor_levels(df,col_to_remap:str,remap:dict):
    # old_name is each value in the column to be remapped
    def mapping(old_name):
        # Lower each value in the column so that it can be checked against remap
        # which is all lowercase
        old_name_lower = old_name.lower()
        # grouping is assigned to the keys of the inputted dict (remap)
        # keywords is assigned to the list of values for that keyword of the inputted dict (remap)
        for grouping, keywords in remap.items():
            # k is the individual values inside the list of values for keywords
            # Checking if any values (k) from our inputted dict are in the values
            # from the column we're remapping, and if so, return grouping, which is the new name 
            if any(k in old_name_lower for k in keywords):
                return grouping
        # Failsafe in case cannot be lowered
        return old_name
    # Applying remap to the column name provided    
    df[col_to_remap] = df[col_to_remap].apply(mapping)
    return df
# %%
# Function to normalize the values of numeric columns between -1 and 1
# datatype: string for the datatype to change, in this case need numeric
def normalize(df,datatype:str):
    # Gathering columns of specified datatype and putting into a list
    columns = list(df.select_dtypes(datatype))
    # Transforming values of specified columns to be between -1 and 1
    # Formula is (x-min)/(max-min)
    df[columns] = MinMaxScaler().fit_transform(df[columns])
    return df
# %%
# Encoding categorical columns to new columns with a binary 1 for existing and 0 for not existing
# datatype: string for datatype to change, in this case need category
def one_hot_encoding(df,datatype:str):
    categories = list(df.select_dtypes(datatype))
    # pandas function to separate categorical columns into individual binary columns
    df_encoded = pd.get_dummies(df,columns=categories)
    return df_encoded
# %%
# Function drop columns based on drop_cols list provided
def drop(df,drop_cols:list):
    clean_df = df.drop(drop_cols,axis=1)
    return clean_df
# %%
# Function to specify a target variable and create a new column as a binary 1 or 0
# target: column name that should be the target variable
# target_f: name for new column where the target variable will be binary
# cutoff: value of 75th percentile to designate where to create the cutoff between the two bins
def new_col_target(df,target:str,target_f:str,cutoff:float):
    df[target_f] = pd.cut(df[target], bins=[-1,cutoff,1],labels=[0,1])
    return df
# %%
# Function to calculate prevalence using the new binary target variable column above
def prevalence(df,target_f:str):
    prev = (df[target_f].value_counts()[1]/len(df[target_f]))
    return prev
# %%
# Splitting data into train, test, and tune datasets, which are all from the binary target variable column
# first_split_size: integer to decide how many values to be put into the train variable vs. the test variable
# second_split_prop: proportion to decide how to proportionally split up values between tune and test (in this case .5)
def split(df,target_f: str,first_split_size:int,second_split_prop:float):
    df = df.dropna(subset=[target_f])
    train,test = train_test_split(
        df,
        train_size = first_split_size,
        stratify = df[target_f]
    )
    tune,test = train_test_split(
        test,
        train_size = second_split_prop,
        stratify=test[target_f]
    )
    return (train, test, tune)
#%%
# TESTING BELOW HERE.
collegedf = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")
# %%
df = data_cleaning(collegedf,cat=['state','basic'],bool_col=["hbcu","flagship"])
#%%
region_map = {
    "Midwest": [
        'texas','oklahoma','kansas','nebraska','south dakota','north dakota',
        'minnesota','wisconsin','iowa','missouri','arkansas','louisiana',
        'illinois','indiana','ohio','michigan'
    ],
    "Northeast": [
        'pennsylvania','new york','new jersey','connecticut','rhode island',
        'massachusetts','vermont','new hampshire','maine'
    ],
    "Pacific": [
        'california','oregon','washington','alaska','hawaii'
    ],
    "Southeast": [
        'kentucky','tennessee','virginia','west virginia','maryland',
        'delaware','north carolina','south carolina','mississippi',
        'alabama','georgia','florida'
    ],
    "West": [
        'montana','idaho','wyoming','nevada','utah',
        'colorado','arizona','new mexico'
    ]
}
# %%
df = factor_levels(df,col_to_remap="state",remap=region_map)
# %%
basic_map = {
    "Associates": ["associate"],
    "Masters": ["master"],
    "Theological": ["Theological"],
    "Baccalaureate": ["baccalaureate","bachelor"],
    "Art": ["art"],
    "Research": ["research"],
    "Other": ["other","schools of","tribal","not applicable"]
}
# %%
df = factor_levels(df,col_to_remap="basic",remap=basic_map)
#%%
df = normalize(df,'number')
# %%
df = one_hot_encoding(df,'category')
# %%
df = drop(df,['index','unitid','chronname','site','long_x','lat_y','vsa_year','vsa_enroll_elsewhere_after4_first',
              'vsa_grad_after4_first','vsa_grad_elsewhere_after4_first','vsa_enroll_after4_first','city','vsa_enroll_after6_first',
              'vsa_grad_after6_first','vsa_grad_elsewhere_after6_first','vsa_enroll_elsewhere_after6_first','med_sat_value',
              'vsa_grad_after4_transfer','vsa_grad_elsewhere_after4_transfer','vsa_enroll_after4_transfer','med_sat_percentile',
              'vsa_enroll_elsewhere_after4_transfer','vsa_grad_after6_transfer','vsa_grad_elsewhere_after6_transfer',
              'vsa_enroll_after6_transfer','vsa_enroll_elsewhere_after6_transfer','similar','counted_pct','nicknames'])
# %%
df = new_col_target(df,"awards_per_state_value","awards_per_state_value_f",0.37037)
# %%
preval = prevalence(df,"awards_per_state_value_f")
print(f"Prevalence: {preval}")
# %%
train,test,tune = split(df, "awards_per_state_value_f",2778,.5)
# %%
print(f"Train: {train.shape}, Test: {test.shape}, Tune: {tune.shape}")
# %%
"""Complete!"""