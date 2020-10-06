# -- coding: utf-8 --
"""
Created on Thu Jun 18 08:54:38 2020

@author: Kavithav
"""

import pandas as pd
# import numpy as np

#Importing Data
Bank = pd.read_csv("D:/Data science/Module 8 - LogRegression/Affairs (1)\Bank-full.csv")

### Splitting the data into train and test data
Bank.drop(Bank.columns[[8,9,10,11,13,14,15]], axis=1, inplace=True)
Bank.isnull()
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(Bank, test_size = 0.3) # 30% test data

# Model building

import statsmodels.formula.api as sm
logit_model = sm.logit('y ~ age+default+balance+housing+loan+campaign', data = train_data).fit()

#summary
logit_model.summary()

## Evaluation of the model
predict_test = logit_model.predict(pd.DataFrame(test_data[['age','default','balance','housing','loan','campaign']]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cnf_test_matrix = confusion_matrix(test_data['y'], predict_test > 0.5 )
cnf_test_matrix

print(accuracy_score(test_data.loan, predict_test > 0.5))

## Error on train data
predict_train = logit_model.predict(pd.DataFrame(train_data[['age','default','balance','housing','loan','campaign']]))

cnf_train_matrix = confusion_matrix(train_data['loan'], predict_train > 0.5 )
cnf_train_matrix

print(accuracy_score(train_data.loan, predict_train > 0.5))
