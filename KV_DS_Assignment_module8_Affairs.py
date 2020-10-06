# -- coding: utf-8 --
"""
Created on Thu Jun 18 08:54:38 2020

@author: Kavithav
"""

import pandas as pd
# import numpy as np

#Importing Data
Affairs = pd.read_csv("D:/Data science/Module 8 - LogRegression/Affairs (1)\Affairs (1).csv")

### Splitting the data into train and test data
Affairs.drop(Affairs.columns[[0]], axis=1, inplace=True)
Affairs.isnull()
Affairs['naffairs'] = Affairs.naffairs.apply(lambda x: 1 if x != 0 else 0)
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(Affairs, test_size = 0.3) # 30% test data
Affairs.columns
# Model building

import statsmodels.formula.api as sm
logit_model = sm.logit('naffairs ~ vryunhap+unhap+avgmarr+hapavg+vryhap+antirel+notrel+slghtrel+smerel+vryrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5+yrsmarr6', data = train_data).fit()

#summary
logit_model.summary()

## Evaluation of the model
predict_test = logit_model.predict(pd.DataFrame(test_data[['kids', 'vryunhap', 'unhap', 'avgmarr', 'hapavg', 'vryhap','antirel', 'notrel', 'slghtrel', 'smerel', 'vryrel', 'yrsmarr1','yrsmarr2', 'yrsmarr3', 'yrsmarr4', 'yrsmarr5', 'yrsmarr6']]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cnf_test_matrix = confusion_matrix(test_data['naffairs'], predict_test > 0.5 )
cnf_test_matrix

print(accuracy_score(test_data.naffairs, predict_test > 0.5))

## Error on train data
predict_train = logit_model.predict(pd.DataFrame(train_data[['kids', 'vryunhap', 'unhap', 'avgmarr', 'hapavg', 'vryhap','antirel', 'notrel', 'slghtrel', 'smerel', 'vryrel', 'yrsmarr1','yrsmarr2', 'yrsmarr3', 'yrsmarr4', 'yrsmarr5', 'yrsmarr6']]))
cnf_train_matrix = confusion_matrix(train_data['naffairs'], predict_train > 0.5 )
cnf_train_matrix

print(accuracy_score(train_data.naffairs, predict_train > 0.5))
