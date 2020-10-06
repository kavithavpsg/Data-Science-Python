# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 06:29:11 2020

@author: KavithaV
"""
import pandas as pd
import numpy as np

##Predict profit of a start up company
start_ups = pd.read_csv("D:/Data science/Module 7-MLR/Assignment/50_Startups (2)/50_Startups (2).csv")
start_ups.columns = "RD","Admin","Mkt","State","Profit"
start_ups.describe()
startups_dummies = pd.get_dummies(start_ups, columns = ['State'])
startups_dummies.columns = "RD","Admin","Mkt","Profit","State_California","State_Florida","State_Newyork"
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startups_dummies.iloc[:,:])
                             
# Correlation matrix 
startups_dummies.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit~RD+Mkt+Admin+State_California+State_Newyork+State_Florida',data=startups_dummies).fit() # regression model

# Summary
ml1.summary()

# calculating VIF's values of independent variables
rsq_RD = smf.ols('RD~Mkt+Admin+State_California+State_Newyork+State_Florida',data=startups_dummies).fit().rsquared  
vif_RD = 1/(1-rsq_RD) 

rsq_Mkt = smf.ols('Mkt~RD+Admin+State_California+State_Newyork+State_Florida',data=startups_dummies).fit().rsquared  
vif_Mkt = 1/(1-rsq_Mkt)

rsq_Admin = smf.ols('Admin~RD+Mkt+State_California+State_Newyork+State_Florida',data=startups_dummies).fit().rsquared  
vif_Admin = 1/(1-rsq_Admin) 

#rsq_State = smf.ols('State~RD+Mkt+Admin',data=start_ups).fit().rsquared  
#vif_State = 1/(1-rsq_State) 



# Storing vif values in a data frame
d1 = {'Variables':['RD','Mkt','Admin'],'VIF':[vif_RD,vif_Mkt,vif_Admin]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# final model
final_ml= smf.ols('Profit~Mkt+Admin+RD+State_California+State_Newyork+State_Florida',data = startups_dummies).fit()
final_ml.summary()


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startups_train,startups_test  = train_test_split(startups_dummies,test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols('Profit~Admin+Mkt+State_California+State_Newyork+State_Florida',data=startups_train).fit()

# train_data prediction
train_pred = model_train.predict(startups_train)

# train residual values 
train_resid  = train_pred - startups_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 23911.17

train_rmse
# prediction on test data set 
test_pred = model_train.predict(startups_test)

# test residual values 
test_resid  = test_pred - startups_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 27975.87
test_rmse