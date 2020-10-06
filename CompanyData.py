# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:09:24 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
company_data = pd.read_csv("D:/Data science/Module 10 - DT and RF/Assignment/Fraud_check (1)/Company_Data (1).csv")

company_data.isnull().sum()
company_data.dropna()
company_data.columns



#converting into binary
lb = LabelEncoder()
company_data["CompPrice"] = lb.fit_transform(company_data["CompPrice"])
company_data["Income"] = lb.fit_transform(company_data["Income"])
company_data["Advertising"] = lb.fit_transform(company_data["Advertising"])
company_data["Population"] = lb.fit_transform(company_data["Population"])
company_data["Price"] = lb.fit_transform(company_data["Price"])
company_data["ShelveLoc"] = lb.fit_transform(company_data["ShelveLoc"])
company_data["Age"] = lb.fit_transform(company_data["Age"])
company_data["Education"] = lb.fit_transform(company_data["Education"])
company_data["Urban"] = lb.fit_transform(company_data["Urban"])
company_data["US"] = lb.fit_transform(company_data["US"])

company_data['Sales'] = company_data['Sales'].astype('category',copy=False)
#data["default"]=lb.fit_transform(data["default"])

company_data['Sales'].unique()
company_data['Sales'].value_counts()
colnames = list(company_data.columns)
type(company_data.columns)
predictors = colnames[:11]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(company_data, test_size = 0.3)

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

