# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 06:38:06 2020

@author: KavithaV
"""
import pandas as pd
####Predict computer price

computer_data = pd.read_csv("D:/Data science/Module 7-MLR/Assignment/50_Startups (2)/Computer_Data (1).csv")
#Replace categorical with binary values
conv = lambda x:1 if (x == 'yes') else 0
computer_data['cd'] = computer_data['cd'].apply(lambda x: 1 if(x == 'yes') else 0)
computer_data['multi'] = computer_data['multi'].apply(lambda x: 1 if(x == 'yes') else 0)
computer_data['premium'] = computer_data['premium'].apply(lambda x: 1 if(x == 'yes') else 0)

computer_data.describe()
computer_data = computer_data.drop('Unnamed: 0',1)
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(computer_data.iloc[:,:])
                             
# Correlation matrix 
computer_data.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=computer_data).fit() # regression model

# Summary
ml1.summary()

# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed~hd+ram+screen+cd+multi+premium+ads+trend',data=computer_data).fit().rsquared  
vif_speed = 1/(1-rsq_RD) 

rsq_hd = smf.ols('hd~speed+ram+screen+cd+multi+premium+ads+trend',data=computer_data).fit().rsquared  
vif_hd = 1/(1-rsq_hd)

rsq_ram = smf.ols('ram~speed+hd+screen+cd+multi+premium+ads+trend',data=computer_data).fit().rsquared  
vif_ram = 1/(1-rsq_ram) 

rsq_screen = smf.ols('screen~speed+hd+ram+cd+multi+premium+ads+trend',data=computer_data).fit().rsquared  
vif_screen = 1/(1-rsq_screen)

rsq_cd = smf.ols('cd~speed+hd+ram+screen+multi+premium+ads+trend',data=computer_data).fit().rsquared  
vif_cd = 1/(1-rsq_cd)

rsq_multi = smf.ols('multi~speed+hd+ram+cd+screen+premium+ads+trend',data=computer_data).fit().rsquared  
vif_multi = 1/(1-rsq_multi)

rsq_premium = smf.ols('premium~speed+hd+ram+cd+screen+multi+ads+trend',data=computer_data).fit().rsquared  
vif_premium = 1/(1-rsq_premium)

rsq_ads = smf.ols('ads~speed+hd+ram+cd+screen+premium+multi+trend',data=computer_data).fit().rsquared  
vif_ads = 1/(1-rsq_ads)

rsq_trend = smf.ols('trend~speed+hd+ram+cd+screen+premium+ads+multi',data=computer_data).fit().rsquared  
vif_trend = 1/(1-rsq_trend)
 

# Storing vif values in a data frame
d1 = {'Variables':['speed','hd','ram','screen','cd','multi','premium','ads','trend'],'VIF':[vif_speed,vif_hd,vif_ram,vif_screen,vif_cd,vif_multi,vif_premium,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# As hd is having higher VIF value, we are not going to include this prediction model
# final model
final_ml= ml1 = smf.ols('price~speed+ram+screen+cd+multi+premium+ads+trend',data=computer_data).fit()
final_ml.summary()


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
computerdata_train,computerdata_test  = train_test_split(computer_data,test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols('price~speed+ram+screen+cd+multi+premium+ads+trend',data=computer_data).fit()

# train_data prediction
train_pred = model_train.predict(computerdata_train)

# train residual values 
train_resid  = train_pred - computerdata_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 285.52

train_rmse
# prediction on test data set 
test_pred = model_train.predict(computerdata_test)

# test residual values 
test_resid  = test_pred - computerdata_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 285.52
test_rmse