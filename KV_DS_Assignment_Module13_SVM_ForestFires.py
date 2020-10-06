# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 06:45:46 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Loading the data set
forestfires_data = pd.read_csv("D:/Data science/Module 13-SVM/SalaryData_Train (3)/forestfires (3).csv")
	
forestfires_data.isnull().sum()

print(forestfires_data.dtypes)
#Encoding categorical values

obj_df = forestfires_data.select_dtypes(include=['object']).copy()
for value in ['month', 'day']:
    forestfires_data[value] = forestfires_data[value].astype('category')
    forestfires_data[value] = forestfires_data[value].cat.codes

X = np.array(forestfires_data.iloc[:,0:29]) # Predictors 
Y = np.array(forestfires_data['size_category']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(X_train,Y_train)
pred_test_linear = model_linear.predict(X_test)

np.mean(pred_test_linear==Y_test)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train,Y_train)
pred_test_rbf = model_rbf.predict(X_test)

np.mean(pred_test_rbf==Y_test)