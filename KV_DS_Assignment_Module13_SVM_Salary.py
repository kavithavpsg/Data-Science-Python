# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 06:45:46 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Loading the data set
salary_data_train = pd.read_csv("D:/Data science/Module 12-Bayesian/Assignments/Bayesian Classifier_Assignments/SalaryData_Train.csv")
salary_data_test = pd.read_csv("D:/Data science/Module 12-Bayesian/Assignments/Bayesian Classifier_Assignments/SalaryData_Test.csv")

	
salary_data_train.isnull().sum()
for value in ['workclass', 'education','maritalstatus', 'occupation','relationship','race', 'sex','native', 'Salary']:
    print( sum(salary_data_train[value] == '?'))

print(salary_data_train.dtypes)
#Encoding categorical values

obj_df = salary_data_train.select_dtypes(include=['object']).copy()
for value in ['workclass', 'education','maritalstatus', 'occupation','relationship','race', 'sex','native']:
    salary_data_train[value] = salary_data_train[value].astype('category')
    salary_data_train[value] = salary_data_train[value].cat.codes

num_features = ['age', 'workclass', 'education', 'educationno',
                'maritalstatus', 'occupation', 'relationship', 'race',
                'sex', 'capitalgain', 'capitalloss', 'hoursperweek',
                'native']
 
scaled_features = {}
for each in num_features:
    mean, std = salary_data_train[each].mean(), salary_data_train[each].std()
    scaled_features[each] = [mean, std]
    salary_data_train.loc[:, each] = (salary_data_train[each] - mean)/std


for value in ['workclass', 'education','maritalstatus', 'occupation','relationship','race', 'sex','native']:
    salary_data_test[value] = salary_data_test[value].astype('category')
    salary_data_test[value] = salary_data_test[value].cat.codes

num_features = ['age', 'workclass', 'education', 'educationno',
                'maritalstatus', 'occupation', 'relationship', 'race',
                'sex', 'capitalgain', 'capitalloss', 'hoursperweek',
                'native']
 
scaled_features = {}
for each in num_features:
    mean, std = salary_data_test[each].mean(), salary_data_test[each].std()
    scaled_features[each] = [mean, std]
    salary_data_test.loc[:, each] = (salary_data_test[each] - mean)/std    
train_X = salary_data_train.iloc[:,:12]
train_y = salary_data_train.iloc[:,13]
test_X  = salary_data_test.iloc[:,:12]
test_y  = salary_data_test.iloc[:,13]

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)