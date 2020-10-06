# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:09:24 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np
Diabetes = pd.read_csv("D:/Data science/Module 10 - DT and RF/Assignment/Fraud_check (1)/Diabetes (1).csv")
Diabetes.columns = Diabetes.columns.str.replace(' ', '')
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
Diabetes_n = norm_func(Diabetes.iloc[:,0:8])
Diabetes_n.describe()

X = np.array(Diabetes_n.iloc[:,0:8]) # Predictors 
Y = np.array(Diabetes['Classvariable']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

oob_score=True,

rf.fit(X_train, Y_train) # Fitting RandomForestClassifier model from sklearn.ensemble  
pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score

pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions']) 
print(accuracy_score(Y_test, pred))

# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2

# train accuracy 
train_acc2 = np.mean(rf.predict(X_train)==Y_train)
train_acc2

