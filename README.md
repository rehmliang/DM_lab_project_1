# DM_lab_project_1
This is  a github page for DM_lab Project 1.
--maintained by Rui Liang, Hamza Ben Slimen

#### Dataset

There are 3 datasets used for this experiment: Census(adult), german, singles

We first did exploratory analyssis for the datasets
![Alt text]( "optional title")

The analysis for each dataset will be stored in file'dataset exploratory analysis' as .ipynb

#### dataset preprocess:
deal with missing value, i.e, in adult dataset, there ara missing values only in categorical features, and the number will not affect the prediction,so discard all missing value  \
categorical feature --> numerical feature
\
train-test split: 2/3  1/3
\
reindex the protected feature(sensitive) 

#### compute statistical parity (SP) of the dataset

#### implement SDB on logistic regression

#### implement SDB on SVM

#### implement SDB on AdaBoost


#### evaluation metrics: label error, bias(SP), RRB(resilience to random bias)

