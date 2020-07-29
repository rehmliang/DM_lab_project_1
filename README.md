# DM_lab_project_1
This is  a github page for DM_lab Project 1.
--maintained by Rui Liang, Hamza Ben Slimen

#### Dataset

We used 2 datasets for this experiment: Census(adult), german, the same as in the paper

We first did exploratory analyssis for the datasets 

i.e. overview of the adult dataset:
![Alt text]( https://github.com/rehmliang/DM_lab_project_1/blob/master/dataset%20exploratory%20analysis/figures%20for%20adult/head.png)

the proportion of poctected instances labeled positively: 
![Alt text](https://github.com/rehmliang/DM_lab_project_1/blob/master/dataset%20exploratory%20analysis/figures%20for%20adult/sex.png)

As can be seen, we found that the 'sex' is not the only sensitive attribute in adult dataset, 'race' can be also considered as sensitive attribute
![Alt text](https://github.com/rehmliang/DM_lab_project_1/blob/master/dataset%20exploratory%20analysis/figures%20for%20adult/race.png) 

The complete analysis for each dataset will be stored in file'dataset exploratory analysis' as .ipynb

#### dataset preprocess:
First, we deal with missing value, i.e, in adult dataset, there ara missing values only in categorical features, and the number will not affect the prediction,so discard all missing value  

categorical feature --> numerical feature 

train-test split: 2/3  1/3 

reindex the protected feature(sensitive) i.e, in adult dataset, set 'sex' in 2nd place

#### Main Idea of the proposed method
Use the confidence score based on boosting hypothesis to find the optimal error decision boundary shift for protected group that achieves statistical Parity. \\
It follows the logic: if the confidence is lower,  it's more possible the data point is misclassified.
we found the data points with small confidence, and flip their label to achieve statistical parity. 

i.e. this figure shows the confidence score of protected group and others of Adaboost:
![Alt text](https://github.com/rehmliang/DM_lab_project_1/blob/master/method/plots/boost%20adult%20hist.png) 

The SDB method generalize also on Logistic Regression and SVM:

#### SDB on logistic regression
deﬁne the conﬁdence of logistic regression simply as the value that the classiﬁer takes before rounding
#### SDB on SVM
deﬁne the conﬁdence as the distance of a point from the separating hyperplane

#### Experiments and Results
We reproduce the proposed method in the paper and use the same setup (run the method for each dataset 10 times and take the average, the whole process takes about 12hours)
The experimental results are reported in the method/experiment-SDB.py and method/experiment-SDB-race.py , the figures are in  method/plots.
#### Evaluation metrics: label error, bias(Statistical Parity), RRB(resilience to random bias)
#### Conclusion
## The method achieved fairness-accuracy tradeoff, which can be controlled after training and makes it a fast and transparent method. 
## Adjusting the decision boundary of learner based on the confidence scores of the misclassified instances.

## The SDB method can only deal with single protected attribute, if there are more than one sensitive attribute in datasets,  cannot handle all discriminations simultaneously.

## It could be combined with other pre-processing or learner-specific in-processing method

## It doesn’t consider the class-imbalance



