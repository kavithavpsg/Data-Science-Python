# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:09:24 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
Fraud_check = pd.read_csv("D:/Data science/Module 10 - DT and RF/Assignment/Fraud_check (1)/Fraud_check (1).csv")
Fraud_check.columns = Fraud_check.columns.str.replace('.', '')
Fraud_check.isnull().sum()
Fraud_check.dropna()
Fraud_check.columns



#converting into binary
lb = LabelEncoder()
Fraud_check["Undergrad"] = lb.fit_transform(Fraud_check["Undergrad"])
Fraud_check["MaritalStatus"] = lb.fit_transform(Fraud_check["MaritalStatus"])
Fraud_check["TaxableIncome"] = lb.fit_transform(Fraud_check["TaxableIncome"])
Fraud_check["CityPopulation"] = lb.fit_transform(Fraud_check["CityPopulation"])
Fraud_check["WorkExperience"] = lb.fit_transform(Fraud_check["WorkExperience"])
Fraud_check["Urban"] = lb.fit_transform(Fraud_check["Urban"])
for index, rows in Fraud_check.iterrows():
         if(rows["TaxableIncome"] <= 300 ):
             Fraud_check.loc[index,"Riskcategory"] = 'Risky' 
         else: Fraud_check.loc[index,"Riskcategory"] = 'Good'
Fraud_check['Riskcategory'] = pd.Categorical(Fraud_check['Riskcategory'])
Fraud_check.dtypes

#data["default"]=lb.fit_transform(data["default"])

Fraud_check['Riskcategory'].unique()
Fraud_check['Riskcategory'].value_counts()
colnames = list(Fraud_check.columns)
type(Fraud_check.columns)
predictors = colnames[:6]
target = colnames[6]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraud_check, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

