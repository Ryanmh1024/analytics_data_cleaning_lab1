#%%
# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling

#%%
jobdf = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
jobdf.head()
#%%
"""Question: Do higher MBA Percentages correlate to Job Placement?"""
#%%
"""Step Two - Job Dataset
Independent Business Metric: Assuming we can attribute Job Placement to MBA percentages,
can we predict Job Placement in the future based on MBA percentage?
"""
#%%
# Data Preparation:
jobdf.info()
#%%
"""Correct variable type/class as needed - change to categorical columns
"""
# Putting column names in a list
category_list = list(jobdf.select_dtypes('str'))
# Converting columns to categorical
jobdf[category_list] = jobdf[category_list].astype('category')
#%%
"""Fix factor levels: Check that categorical variable don't have too many groups
"""
#%%
# Gender - Male or Female:
jobdf[category_list[0]].value_counts()
#%%
# ssc_b - Secondary Education Board of Education; Central or Others:
jobdf[category_list[1]].value_counts()
#%%
# hsc_b - Higher Secondary Education Board of Education; Central or Others:
jobdf[category_list[2]].value_counts()
#%%
# hsc_s - Specialization in Higher Secondary Education; Commerce, Science, or Arts:
jobdf[category_list[3]].value_counts()
#%%
# degree_t - Undergraduate Degree Type; Comm&Mgmt, Sci&Tech, or Others:
jobdf[category_list[4]].value_counts()
#%%
# workex - Work Experience; Yes or No:
jobdf[category_list[5]].value_counts()
#%%
# specialisation - Speciality in their field; Mkt&Fin or Mkt&HR:
jobdf[category_list[6]].value_counts()
#%%
# status - Job Placement status; Placed or Not Placed:
jobdf[category_list[7]].value_counts()
# All categorical columns are well grouped into 2 or 3 groups already!
#%%
"""Normalize the continuous data
"""
#%%
# Getting all numeric columns
numeric = list(jobdf.select_dtypes('number'))
# Normalizing all numeric columns by putting their values between -1 and 1
# Formula: (x-min)/(max-min)
jobdf[numeric] = MinMaxScaler().fit_transform(jobdf[numeric])

#%%
"""One-hot encoding factor variables to be put through machine learning
"""
#%%
# Use categories list from earlier for categorical columns
jobdf_encoded = pd.get_dummies(jobdf,columns=category_list)
#%%
"""Dropping variables that are not necessary for computation
"""
#%%
jobdf_clean = jobdf_encoded.drop(['sl_no','ssc_b_Central','ssc_b_Others','hsc_b_Central','hsc_b_Others'],axis=1)
#%%
# Choosing MBA percentage to be target variable as specified in my question earlier
# Printing boxplot of target variable to see distribution of values
print(jobdf_clean.boxplot(column='mba_p', vert=False, grid=False))
#%%
# Checking statistical values for target variable
print(jobdf_clean.mba_p.describe())
# 75th Percentile value is 0.563906
#%%
# Creating a new column that groups the target variable into two bins above or below the 75th percentile
jobdf_clean['mba_p_f'] = pd.cut(jobdf_clean.mba_p,
                                bins=[-1,0.5639,1],
                                labels=[0,1])
#%%
# Using prevalence formula with numerator as True values (1's)
# and denominator as total number of values in column
prevalence = (jobdf_clean['mba_p_f'].value_counts()[1]/len(jobdf_clean['mba_p_f']))
# This is the accuracy that our model should be able to beat
print(f"Prevalence: {prevalence}")
#%%
# Splitting data for target variable into train, test, and tune
train, test = train_test_split(
    jobdf_clean,
    train_size=161,
    stratify=jobdf_clean['mba_p_f']
)
tune, test = train_test_split(
    test,
    train_size=.5,
    stratify=test['mba_p_f']
)
#%%
print(f"Training set shape: {train.shape}")
print(f"Testing set shape: {test.shape}")
print(f"Tuning set shape: {tune.shape}")
jobdf_clean
#%%
# The pattern for salary NaN data is simply whether they have been
# placed into a job or not, because the people with Placement have a salary,
# whereas the people without a job placement have no salary because they have no job
#%%
"""
Step 3: What do your instincts tell you about the data? 
Can you address your problem, what areas/items are you worried about?

My instincts are telling me that the MBA percentage is not necessarily the strongest
indicator of job placement because there are so many True values for job placement
and not a correlating amount of 1's for binary MBA percentile above or below 75th percentile,
but there are other factors including percentile in higher secondary education and secondary
education that could lead to higher Job Placement.
"""