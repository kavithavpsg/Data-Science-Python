# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:32:17 2020

@author: KavithaV
"""
########################## Neural Network for predicting continuous values ###############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
a=100

# Reading data 
startups = pd.read_csv("D:/Data science/Module 14 - NN/50_Startups (3)/50_Startups (3).csv")
startups.head()

startups.describe()
#Check for outliers in data
plt.boxplot(startups['R&D Spend'])
plt.boxplot(startups['Administration'])
plt.boxplot(startups['Marketing Spend'])
plt.boxplot(startups['Profit']) #One outlier in Profit column

#Standardization of input data
startups['State'] = startups['State'].astype('category')
startups['State'] = startups['State'].cat.codes


num_features = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
 
scaled_features = {}
for each in num_features:
    mean, std = startups[each].mean(), startups[each].std()
    scaled_features[each] = [mean, std]
    startups.loc[:, each] = (startups[each] - mean)/std

from sklearn.model_selection import train_test_split

X = startups.drop(["Profit"],axis=1)
Y = startups["Profit"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

cont_model = Sequential()
cont_model.add(Dense(50, input_dim=4, activation="relu"))
cont_model.add(Dense(250, activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mse"])

model = cont_model
model.fit(np.array(X_train), np.array(y_train), epochs=20)

# On Test dataset
pred = model.predict(np.array(X_test))
pred = pd.Series([i[0] for i in pred])

# Accuracy
np.corrcoef(pred, y_test)

layerCount = len(model.layers)
layerCount

# On Train dataset
pred_train = model.predict(np.array(X_train))
pred_train = pd.Series([i[0] for i in pred_train])

np.corrcoef(pred_train, y_train) #this is just because some model's count the input layer and others don't



