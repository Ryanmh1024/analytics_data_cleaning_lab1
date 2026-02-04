#%%
# Importing required packages
import pandas as pd
import numpy as np

# %%
collegedf = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")
collegedf.head()
"""
Generic Question: What metrics lead to higher SAT values?
"""

# %%
jobdf = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
jobdf.head()
""" 
Question: Do higher MBA Percentages correlate to Job Placement?
"""

#%%
"""Step Two - College Dataset
Independent Business Metric: Assuming we can find what metrics indicate higher SAT,
can we predict the metrics next year that will indicate more SAT?
"""
# Data Preparation:
# %%
collegedf.info()

#%%
# Fixing boolean columns in hbcu and flagship.
collegedf["hbcu"] = collegedf["hbcu"].apply(lambda x: 1 if x == 'X' else 0)
collegedf["flagship"] = collegedf["flagship"].apply(lambda x: 1 if x == 'X' else 0)

#%%
"""
Correct variable type/class as needed - making categorical variables. 
"""
# Putting the column names that need to be categorical into a variable
Column_index_list = ["chronname","city","state","level","control","basic","site"]
# Converting columns necessary to categorical for computation later.
collegedf[Column_index_list]= collegedf[Column_index_list].astype('category')

# %%
# Checking that the code above did its job and the intended columns are now categorical.
collegedf.dtypes

#%%
"""
Fix factor levels: Check that categorical variables don't have too many groups.
"""
collegedf.city.value_counts()[0:10]

#%%
# Fixing city category to only be top

#%%
# Prevalence is calculating the percentage of targets that pass (0 or 1) (ballpark 10-20%)
# Usually care about modeling something that occurs a small percentage of the time.