# Data Cleaning Lab 1 Instructions

Goal: Build two data prep pipelines using different datasets to get practice with data preparation and question building. In doing so, create a new github repo for your work. Think of this as a stand alone project that requires the creation of a workspace and repository. In the repo, it is likely best to create three files. One for the actual assignment details (this file), a second (python file) to answer questions 1-3 and a third (python file) for question 4.

**Step One**: Review the two datasets and brainstorm problems that could be addressed with the dataset. Identify a question for each dataset.

[College Completion Dataset](https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv) 

* [College Completion Data Dictionary + Data](https://www.kaggle.com/datasets/thedevastator/boost-student-success-with-college-completion-da/data)

[Job Placement Dataset](https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv)

* [Job Placement Data Dictionary of sorts](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement/discussion/280612)

**Step Two**: Work through the steps outlined in the examples to include the following elements:

* Write a generic question that this dataset could address.
* What is an independent Business Metric for your problem? Think about the case study examples we have discussined in class.
* Data Preparation:
  * Correct variable type/class as needed
  * Collapse factor levels as needed
  * One-hot encoding factor variables
  * normalize the continuous variables
  * Drop unneeded variables
  * Create target variable if needed
  * Calculate the prevalence of the target variable
  * Create the nececssary data partitions (Train, Tune, Test)

**Step Three**: What do your instincts tell you about the data? Can it address your problem, what areas/items are you worried about?

**Step Four**: Create functions for your two pipelines that produces the train and test datasets. The end result should be a series of functions that can be called to produce the train and test datasets for each of your two problems that includes all the data prep steps you took. This is essentially creating a DAG for your data prep steps. Imagine you will need to do this for multiple problems in the future so creating functions that can be reused is important. You don't need to create one full pipeline function that does everything, but rather a series of smaller functions that can be called in sequence to produce the final datasets. Use your judgment on how to break up the functions.