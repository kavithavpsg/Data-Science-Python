# -*- coding: utf-8 -*-
"""
Created on Sun May 24 06:33:36 2020

@author: KavithaV
"""
import pandas as pd
import numpy as np
###Predict weight based on calories
calories = pd.read_csv("D:/Data science/Module 6 - LR/Assignment/calories_consumed.csv")
calories.columns = "Weight","Cal"
import matplotlib.pylab as plt #for different types of plots

plt.scatter(x=calories['Cal'], y=calories['Weight'],color='green')# Scatter plot

np.corrcoef(calories.Cal, calories.Weight) #correlation

help(np.corrcoef)

import statsmodels.formula.api as smf
help(smf.ols)
model = smf.ols('Weight ~ Cal', data=calories).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(calories['Cal']))
pred1
print (model.conf_int(0.01)) # 99% confidence interval

res = calories.Weight - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)


######### Model building on Transformed Data

# Log Transformation
# x = log(waist); y = at
plt.scatter(x=np.log(calories['Cal']),y=calories['Weight'],color='brown')
np.corrcoef(np.log(calories.Cal), calories.Weight) #correlation

model2 = smf.ols('Weight ~ np.log(Cal)',data=calories).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calories['Cal']))
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = calories.Weight - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)

# Exponential transformation
plt.scatter(x=calories['Cal'], y=np.log(calories['Weight']),color='orange')

np.corrcoef(calories.Cal, np.log(calories.Weight)) #correlation

model3 = smf.ols('np.log(Weight) ~ Cal',data=calories).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(calories['Cal']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = calories.Weight - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)

##Delivery_time -> Predict delivery time using sorting time 
DT = pd.read_csv("D:/Data science/Module 6 - LR/Assignment/delivery_time.csv")
DT
DT.columns = "DTime","STime"
import matplotlib.pylab as plt #for different types of plots

plt.scatter(x=DT['STime'], y=DT['DTime'],color='green')# Scatter plot

np.corrcoef(DT.STime, DT.DTime) #correlation

import statsmodels.formula.api as smf

model = smf.ols('DTime ~ STime', data=DT).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(DT['STime']))
pred1
print (model.conf_int(0.01)) # 99% confidence interval

res = DT.DTime - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)


######### Model building on Transformed Data

# Log Transformation
# x = log(waist); y = at
plt.scatter(x=np.log(DT['STime']),y=DT['DTime'],color='brown')
np.corrcoef(np.log(DT.STime), DT.DTime) #correlation

model2 = smf.ols('DTime ~ np.log(STime)',data=DT).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(DT['STime']))
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = DT.DTime - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)

# Exponential transformation
plt.scatter(x=DT['STime'], y=np.log(DT['DTime']),color='orange')

np.corrcoef(DT.STime, np.log(DT.DTime)) #correlation

model3 = smf.ols('np.log(DTime) ~ STime',data=DT).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(DT['STime']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = DT.DTime - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)


##Delivery_time -> Predict delivery time using sorting time 
Emp = pd.read_csv("D:/Data science/Module 6 - LR/Assignment/emp_data.csv")

import matplotlib.pylab as plt #for different types of plots

plt.scatter(x=Emp['Salary_hike'], y=Emp['Churn_out_rate'],color='green')# Scatter plot

np.corrcoef(Emp.Salary_hike, Emp.Churn_out_rate) #correlation

import statsmodels.formula.api as smf

model = smf.ols('Churn_out_rate ~ Salary_hike', data=Emp).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(Emp['Salary_hike']))
pred1
print (model.conf_int(0.01)) # 99% confidence interval

res = Emp.Churn_out_rate - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)


######### Model building on Transformed Data

# Log Transformation
# x = log(waist); y = at
plt.scatter(x=np.log(Emp['Salary_hike']),y=Emp['Churn_out_rate'],color='brown')
np.corrcoef(np.log(Emp.Salary_hike), Emp.Churn_out_rate) #correlation

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)',data=Emp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(Emp['Salary_hike']))
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = Emp.Churn_out_rate - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)

# Exponential transformation
plt.scatter(x=Emp['Salary_hike'], y=np.log(Emp['Churn_out_rate']),color='orange')

np.corrcoef(Emp.Salary_hike, np.log(Emp.Churn_out_rate)) #correlation

model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike',data=Emp).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(Emp['Salary_hike']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = Emp.Churn_out_rate - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)

salary = pd.read_csv("D:/Data science/Module 6 - LR/Assignment/Salary_Data.csv")
salary
import matplotlib.pylab as plt #for different types of plots

plt.scatter(x=salary['YearsExperience'], y=salary['Salary'],color='green')# Scatter plot

np.corrcoef(salary.YearsExperience, salary.Salary) #correlation

import statsmodels.formula.api as smf

model = smf.ols('Salary ~ YearsExperience', data=salary).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(salary['YearsExperience']))
pred1
print (model.conf_int(0.01)) # 99% confidence interval

res = salary.Salary - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)


######### Model building on Transformed Data

# Log Transformation
# x = log(waist); y = at
plt.scatter(x=np.log(salary['YearsExperience']),y=salary['Salary'],color='brown')
np.corrcoef(np.log(salary.YearsExperience), salary.Salary) #correlation

model2 = smf.ols('Salary ~ np.log(YearsExperience)',data=salary).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(salary['YearsExperience']))
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = salary.Salary - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)

# Exponential transformation
plt.scatter(x=salary['YearsExperience'], y=np.log(salary['Salary']),color='orange')

np.corrcoef(salary.YearsExperience, np.log(salary.Salary)) #correlation

model3 = smf.ols('np.log(Salary) ~ YearsExperience',data=salary).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(salary['YearsExperience']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = salary.Salary - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
