#%%
# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling

# %%
collegedf = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")
collegedf.head()
"""
Generic Question: What metrics lead to higher SAT values?
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
Column_index_list = ["state","level","control","basic"]
# Converting columns necessary to categorical for computation later.
collegedf[Column_index_list]= collegedf[Column_index_list].astype('category')

# %%
# Checking that the code above did its job and the intended columns are now categorical.
collegedf.dtypes

#%%
"""
Fix factor levels: Check that categorical variables don't have too many groups.
"""
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
# Inelegant function set to group all other ungrouped colleges into Other
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'other' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'schools of' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'tribal' in x.lower() else x)
collegedf.basic = collegedf.basic.apply(lambda x: "Other" if 'not applicable' in x.lower() else x)

#%%
collegedf.basic.value_counts()
#%%
"""
Standardize and Normalize the continuous data
"""
#%%
# Gathering all numeric columns
continuous = list(collegedf.select_dtypes('number'))
#%%
# Bounds all values between 0 and 1 by Min-Max normalization
# Formula = (x-min)/(max-min)
collegedf[continuous] = MinMaxScaler().fit_transform(collegedf[continuous])

#%%
# Need to verify that scaling did not change the density function of original dataframe
collegedf.awards_per_state_value.plot.density()
#%%
# Make sure to check the right column, indicing with 2 displays the awards_per_value column
pd.DataFrame(collegedf_normalized)[8].plot.density()
# Success! Both densities are the same!
#%%
"""
One-hot encoding factor variables to be put through machine learning
"""
#%%
# Getting all categorical columns and putting them into a list
categories = list(collegedf.select_dtypes('category'))
# Get dummies encodes the categorical variables by adding a new column 
# for each group of a category and assigning it to 0 or 1 for false or true
collegedf_1h = pd.get_dummies(collegedf, columns = categories)
collegedf_1h

#%%
"""
Dropping variables that are not necessary for computation.
"""
#%%
collegedf_clean = collegedf_1h.drop(['index','unitid','chronname','site','long_x','lat_y','vsa_year','vsa_enroll_elsewhere_after4_first',
                                     'vsa_grad_after4_first','vsa_grad_elsewhere_after4_first','vsa_enroll_after4_first','city','vsa_enroll_after6_first',
                                     'vsa_grad_after6_first','vsa_grad_elsewhere_after6_first','vsa_enroll_elsewhere_after6_first','med_sat_value',
                                     'vsa_grad_after4_transfer','vsa_grad_elsewhere_after4_transfer','vsa_enroll_after4_transfer','med_sat_percentile',
                                     'vsa_enroll_elsewhere_after4_transfer','vsa_grad_after6_transfer','vsa_grad_elsewhere_after6_transfer',
                                     'vsa_enroll_after6_transfer','vsa_enroll_elsewhere_after6_transfer','similar','counted_pct','nicknames'], axis=1)
collegedf_clean
#%%
# Choosing awards_per_state_value to be target variable because indicative of the number of awards in comparision to the state.
print(collegedf_clean.boxplot(column='awards_per_state_value', vert=False, grid=False))
#%%
# Checking statistical values for this target variable
print(collegedf_clean.awards_per_state_value.describe())
#%%
# Creating a new column that groups the target variable into two bins, above or below the 75th percentile at 0.37
collegedf_clean['awards_per_state_value_f'] = pd.cut(collegedf_clean.awards_per_state_value,
                                    bins=[-1, 0.37037, 1],
                                    labels=[0, 1])
#%%
collegedf_clean
#%%
# Using prevalence formula, where the top is the 1's (indexed by 1 bc those values are second)
# divided by the total number of values in the target variable.
prevalence = (collegedf_clean.awards_per_state_value_f.value_counts()[1] /
              len(collegedf_clean.awards_per_state_value_f))
#%%
# This is the accuracy that our model should be able to beat!
print(f"Prevalence: {prevalence:.2%}")
#%%
# Double checking prevalence calculation
print(collegedf_clean.awards_per_state_value_f.value_counts())
print(f"Manual calculation: 1020/(1020+2778) = {1020/(1020+2778):.4f}")
#%%
# Splitting data for target variable into train and test first
train, test = train_test_split(
    collegedf_clean,
    train_size=2778,
    stratify=collegedf_clean.awards_per_state_value_f
)
#%%
# Size of each dataset for train and test check
print(f"Training set shape: {train.shape}")
print(f"Testing set shape: {test.shape}")
#%%
# Splitting data for target variable again for tuning data
# train_size = .5 means 50% of the data, but an integer represents the number of samples to take for each
tune, test = train_test_split(
    test,
    train_size=.5,
    stratify=test.awards_per_state_value_f
)
#%%
# Checking size of each dataset for training, testing and tuning
print(f"Training set shape: {train.shape}")
print(f"Testing set shape: {test.shape}")
print(f"Tuning set shape: {tune.shape}")

#%%
"""
Step 3: What do your instincts tell you about the data? 
Can you address your problem, what areas/items are you worried about?

My instincts are telling me that this data will not actually help me learn more about the factors
that go into SAT scores because there are no clear variables that can indicate higher SAT scores and
there are too few SAT scores in this dataset to draw any conclusions from.
"""