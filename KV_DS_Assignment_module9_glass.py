# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 06:20:38 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np

glass = pd.read_csv("D:/Data science/Module 9 - KNN/Assignment/Zoo (1)/glass (1).csv")

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
glass_n = norm_func(glass.iloc[:,1:])
glass_n.describe()

X = np.array(glass_n.iloc[:,:]) # Predictors 
Y = np.array(glass['Type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
pd.crosstab(Y_test, pred, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_test, pred))

# error on train data
pred_train = knn.predict(X_train)
pd.crosstab(Y_train, pred_train, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_train, pred_train))
