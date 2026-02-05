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
collegedf.chronname.value_counts()[0:10]
# Not worth altering because there are far too many schools to categorize.
#%%
collegedf.city.value_counts()[20:40]
# Too many cities with high numbers to categorize.

#%%
collegedf.state.value_counts()
# Going to categorize into Midwest, Northeast, Pacific, Southeast, and West.

#%%
midwest = ['Texas','Oklahoma','Kansas','Nebraska','South Dakota','North Dakota',
           'Minnesota','Wisconsin','Iowa','Missouri','Arkansas','Louisiana',
           'Illinois','Indiana','Ohio','Michigan']

northeast = ['Pennsylvania','New York','New Jersey','Connecticut','Rhode Island',
             'Massachusetts','Vermont','New Hampshire','Maine']

pacific = ['California','Oregon','Washington','Alaska','Hawaii']

southeast = ['Kentucky','Tennessee','Virginia','West Virginia','Maryland',
             'Delaware','North Carolina','South Carolina','Mississippi',
             'Alabama','Georgia','Florida']

west = ['Montana','Idaho','Wyoming','Nevada','Utah','Colorado','Arizona','New Mexico']

#%%
# Lambda function explanation:
# lambda x: "Region_Name" if x in region_variable else x
# If the state falls into that region, then it will be renamed to said region
# This is done by checking if it is in the list for that region
# If it is not, then it will simply return x, or its original unaltered nanme
collegedf.state = collegedf.state.apply(lambda x: "Midwest" if x in midwest else x)
collegedf.state = collegedf.state.apply(lambda x: "Northeast" if x in northeast else x)
collegedf.state = collegedf.state.apply(lambda x: "Pacific" if x in pacific else x)
collegedf.state = collegedf.state.apply(lambda x: "Southeast" if x in southeast else x)
collegedf.state = collegedf.state.apply(lambda x: "West" if x in west else x)

#%%
# Check that lambda functions worked as intended.
collegedf.state

#%%
collegedf.level.value_counts()
# Already organized into two categories.

#%%
collegedf.control.value_counts()
# Already well organized into three categories.

#%%
collegedf.basic.value_counts()
# Categorize into specific programs based on Associates, Theological, Masters, etc.

#%%
collegedf.basic.dtype
#%%
associate = ['associate']
master = ["masters"]
theological = ["theological"]
bacclaureate = ["bacclaureate","bachelor"]
art = ["art"]
other = ["Associates","Masters","Theological","Bachelors","Art"]

#%%
# Lambda function explanation:
# lambda x: "Type_of_School" if x in school_variable else x
# If the school falls into that category, then it will be renamed to said school type
# This is done by checking if it is in the list for that school type
# If it is not, then it will simply return x, or its original unaltered nanme
collegedf.basic = collegedf.basic.apply(lambda x: "Associates" if 'associate' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Masters" if 'master' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Theological" if 'theological' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Baccalaureate" if 'baccalaureate' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Art" if 'art' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Research" if 'research' in x.lower() else x)
#%%
# Special inelegant function set to group all other ungrouped colleges into Other
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'other' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'schools of' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'tribal' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'not applicable' in x.lower() else x)

#%%
collegedf.basic.value_counts()
#%%
collegedf.site.value_counts()
# These are websites and don't need to be categorized.
#%%
# Prevalence is calculating the percentage of targets that pass (0 or 1) (ballpark 10-20%)
# Usually care about modeling something that occurs a small percentage of the time.