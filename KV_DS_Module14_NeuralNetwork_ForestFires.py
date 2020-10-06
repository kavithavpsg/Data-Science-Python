# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:32:17 2020

@author: KavithaV
"""
########################## Neural Network for forest fires data ###############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
a=100

# Reading data 
forestfires = pd.read_csv("D:/Data science/Module 14 - NN/50_Startups (3)/forestfires (2).csv")
forestfires.head()

forestfires.describe()
forestfires.isnull().sum()

print(forestfires.dtypes)
#Encoding categorical values

obj_df = forestfires.select_dtypes(include=['object']).copy()
for value in obj_df:
    forestfires[value] = forestfires[value].astype('category')
    forestfires[value] = forestfires[value].cat.codes


#Check for outliers in data
plt.boxplot(forestfires['FFMC'])
plt.boxplot(forestfires['DMC'])
plt.boxplot(forestfires['DC'])
plt.boxplot(forestfires['ISI'])
plt.boxplot(forestfires['temp']) 
plt.boxplot(forestfires['RH'])
plt.boxplot(forestfires['wind'])
plt.boxplot(forestfires['rain'])
plt.boxplot(forestfires['area'])

#data is not normally distributed

#Standardization of input data
names = forestfires.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
forestfires = scaler.fit_transform(forestfires)
forestfires = pd.DataFrame(forestfires, columns=names)

from sklearn.model_selection import train_test_split

X = forestfires.drop(["size_category"],axis=1)
Y = forestfires["size_category"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

cont_model = Sequential()
cont_model.add(Dense(50, input_dim=30, activation="relu"))
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



