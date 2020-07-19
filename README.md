# DM_lab_project_1
This is  a github page for DM_lab Project 1.
--maintained by Rui Liang, Hamza Ben Slimen

#### Dataset

There are 2 datasets used for this experiment: Census(adult), german

We first did exploratory analyssis for the datasets 
i.e. overview of the adult dataset:
![Alt text]( https://github.com/rehmliang/DM_lab_project_1/blob/master/dataset%20exploratory%20analysis/figures%20for%20adult/head.png)

the proportion of poctected instances labeled positively: 
![Alt text](https://github.com/rehmliang/DM_lab_project_1/blob/master/dataset%20exploratory%20analysis/figures%20for%20adult/sex.png)

We found that the 'sex' is not the only sensitive attribute in adult dataset, 'race' can be also considered as sensitive attribute
![Alt text](https://github.com/rehmliang/DM_lab_project_1/blob/master/dataset%20exploratory%20analysis/figures%20for%20adult/race.png) \\

The complete analysis for each dataset will be stored in file'dataset exploratory analysis' as .ipynb

#### dataset preprocess:
deal with missing value, i.e, in adult dataset, there ara missing values only in categorical features, and the number will not affect the prediction,so discard all missing value  \
categorical feature --> numerical feature
\
train-test split: 2/3  1/3
\
reindex the protected feature(sensitive) i.e, in adult dataset, set 'sex' in 2nd place

#### compute statistical parity (SP) of the dataset

#### implement SDB on logistic regression

#### implement SDB on SVM

#### implement SDB on AdaBoost


#### evaluation metrics: label error, bias(SP), RRB(resilience to random bias)

