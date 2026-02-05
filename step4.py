#%%
# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# %%
def data_cleaning(df, cat:list, bool_col:list):
    df[bool_col] = (df[bool_col] == 'X').astype('int')
    df[cat] = df[cat].astype('category')
    return df
#%%
collegedf = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")
# %%
df = data_cleaning(collegedf,cat=['state','basic'],bool_col=["hbcu","flagship"])
df.dtypes
# %%
def factor_levels(df,col_to_remap:str,remap:dict):
    def mapping(old_name):
        old_name_lower = old_name.lower()
        for grouping, keywords in remap.items():
            if any(k in old_name_lower for k in keywords):
                return grouping
        return old_name
    df[col_to_remap] = df[col_to_remap].apply(mapping)
    return df
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
factor_levels(collegedf,col_to_remap="state",remap=region_map)
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
factor_levels(collegedf,col_to_remap="basic",remap=basic_map)
# %%
def normalize(df,datatype:str):
    columns = list(df.select_dtypes(datatype))
    df[columns] = MinMaxScaler().fit_transform(df[columns])
    return df
#%%
normalize(collegedf,'number')
# %%
def one_hot_encoding(df,datatype:str):
    categories = list(df.select_dtypes(datatype))
    df_encoded = pd.get_dummies(df,columns=categories)
    return df_encoded
# %%
one_hot_encoding(collegedf,'category')
# %%
def drop(df,drop_cols:list):
    clean_df = df.drop(drop_cols,axis=1)
    return clean_df

# %%
drop(collegedf,["index","chronname"])
# %%
def new_col_target(df,target:str,target_f:str,cutoff:float):
    df[target_f] = pd.cut(df[target], bins=[-1,cutoff,1],labels=[0,1])
    return df
# %%
new_col_target(collegedf,"awards_per_state_value","awards_per_state_value_f",0.37037)
# %%
def prevalence(df,target_f:str):
    prevalence = (df[target_f].value_counts()[1]/len(df[target_f]))
    return prevalence
# %%
prevalence(collegedf,"awards_per_state_value_f")
# %%
def split(df,target_f: str,first_split_size:int,second_split_prop:float):
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
    return train, test, tune
# %%
train,test,tune = split(collegedf, "awards_per_state_value_f",2778,.5)
# %%
print(f"Train: {train.shape}, Test: {test.shape}, Tune: {tune.shape}")
# %%
