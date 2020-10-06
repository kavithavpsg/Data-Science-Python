# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 06:29:11 2020

@author: KavithaV
"""
import pandas as pd
import numpy as np
####Predict toyoto corolla price

corolla_data = pd.read_excel("D:/Data science/Module 7-MLR/Assignment/50_Startups (2)/ToyotaCorolla (1).xlsx")
corolla_data = corolla_data[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
corolla_data.describe()
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(corolla_data.iloc[:,:])
                             
# Correlation matrix 
corolla_data.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=corolla_data).fit() # regression model

# Summary
ml1.summary()

# calculating VIF's values of independent variables
rsq_Age_08_04 = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax',data=corolla_data).fit().rsquared  
vif_Age_08_04 = 1/(1-rsq_Age_08_04) 

rsq_KM = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax',data=corolla_data).fit().rsquared  
vif_KM = 1/(1-rsq_KM)

rsq_HP = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax',data=corolla_data).fit().rsquared  
vif_HP = 1/(1-rsq_HP) 

rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax',data=corolla_data).fit().rsquared  
vif_cc = 1/(1-rsq_cc) 

rsq_Gears = smf.ols('Gears~Age_08_04+KM+cc+HP+Doors+Quarterly_Tax',data=corolla_data).fit().rsquared  
vif_Gears = 1/(1-rsq_Gears) 

rsq_Doors = smf.ols('Doors~Age_08_04+KM+cc+HP+Gears+Quarterly_Tax',data=corolla_data).fit().rsquared  
vif_Doors = 1/(1-rsq_Doors) 

rsq_QT = smf.ols('Quarterly_Tax~Age_08_04+KM+cc+HP+Gears+Doors',data=corolla_data).fit().rsquared  
vif_QT = 1/(1-rsq_QT) 

# Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax'],'VIF':[vif_Age_08_04,vif_KM,vif_HP,vif_cc,vif_Doors,vif_Gears,vif_QT]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# As KM is having higher VIF value, we are not going to include this prediction model
# final model
final_ml= ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=corolla_data).fit()
final_ml.summary()


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
corolla_data_train,corolla_data_test  = train_test_split(corolla_data,test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=corolla_data_train).fit()

# train_data prediction
train_pred = model_train.predict(corolla_data_train)

# train residual values 
train_resid  = train_pred - corolla_data_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) 

train_rmse
# prediction on test data set 
test_pred = model_train.predict(corolla_data_test)

# test residual values 
test_resid  = test_pred - corolla_data_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse

