# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:23:03 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Loading the data set
salary_data_train = pd.read_csv("D:/Data science/Module 12-Bayesian/Assignments/Bayesian Classifier_Assignments/SalaryData_Train.csv")
salary_data_test = pd.read_csv("D:/Data science/Module 12-Bayesian/Assignments/Bayesian Classifier_Assignments/SalaryData_Test.csv")

	
salary_data_train.isnull().sum()
for value in ['workclass', 'education','maritalstatus', 'occupation','relationship','race', 'sex','native', 'Salary']:
    print( sum(salary_data_train[value] == '?'))
salary_data_train_copy = salary_data_train    
salary_data_train_copy['Sales'] = salary_data_train_copy.Salary.apply(lambda x: 1 if x == ">50K" else 0)
le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(salary_data_train.workclass)
education_cat = le.fit_transform(salary_data_train.education)
marital_cat   = le.fit_transform(salary_data_train.maritalstatus)
occupation_cat = le.fit_transform(salary_data_train.occupation)
relationship_cat = le.fit_transform(salary_data_train.relationship)
race_cat = le.fit_transform(salary_data_train.race)
sex_cat = le.fit_transform(salary_data_train.sex)
native_cat = le.fit_transform(salary_data_train.native)
Salary_cat = le.fit_transform(salary_data_train.Salary)

#initialize the encoded categorical columns
salary_data_train_copy['workclass_cat'] = workclass_cat
salary_data_train_copy['education_cat'] = education_cat
salary_data_train_copy['marital_cat'] = marital_cat
salary_data_train_copy['occupation_cat'] = occupation_cat
salary_data_train_copy['relationship_cat'] = relationship_cat
salary_data_train_copy['race_cat'] = race_cat
salary_data_train_copy['sex_cat'] = sex_cat
salary_data_train_copy['native_country_cat'] = native_cat
salary_data_train_copy['Salary_cat'] = Salary_cat
salary_data_train_copy.isnull().sum()
#drop the old categorical columns
dummy_fields = ['workclass', 'education', 'maritalstatus','occupation', 'relationship', 'race','sex', 'native']
salary_data_train_copy = salary_data_train_copy.drop(dummy_fields, axis = 1)
print(salary_data_train_copy.dtypes)
#Standardize the data
num_features = ['age', 'workclass_cat', 'education_cat', 'educationno',
                'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
                'sex_cat', 'capitalgain', 'capitalloss', 'hoursperweek',
                'native_country_cat']
 
scaled_features = {}
for each in num_features:
    mean, std = salary_data_train_copy[each].mean(), salary_data_train_copy[each].std()
    scaled_features[each] = [mean, std]
    salary_data_train_copy.loc[:, each] = (salary_data_train_copy[each] - mean)/std
#salary_data_train_copy['Salary'] = pd.to_numeric(salary_data_train_copy['Salary'],errors='coerce')
#Split train and test data df.drop(['A'], axis = 1)    
salary_data_train_copy = salary_data_train_copy.drop(['Salary'],axis = 1)    
features = salary_data_train_copy.values[:,:13]
target = salary_data_train_copy.values[:,14]
features_train, features_test, target_train, target_test = train_test_split(features,target, test_size = 0.33, random_state = 10)    

##Naive Bayesian classifier
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)
	
accuracy_score(target_test, target_pred, normalize = True)
