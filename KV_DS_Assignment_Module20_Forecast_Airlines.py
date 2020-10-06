# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 05:49:30 2020

@author: KavithaV
"""

import pandas as pd
airlines = pd.read_excel("D:/Data science/Module 20 - Forecast and Time series/CocaCola_Sales_Rawdata (1)/Airlines Data Set (1).xlsx")
import numpy as np
# Data Preparation
airlines["t"] = np.arange(1,97)

airlines["t_squared"] = airlines["t"]*airlines["t"]
airlines.columns
airlines["log_passengers"] = np.log(airlines["Passengers"])

##Extracting month column from timestamp
airlines['month'] = airlines['Month'].dt.strftime('%b')
# Dummy variables for Seasons
months =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

month_dummies = pd.DataFrame(pd.get_dummies(airlines['month']))
airlines = pd.concat([airlines,month_dummies],axis = 1)

# Partition the data
airlines.iloc[:, 1].plot() # Timeplot
Train = airlines.head(70)
Test = airlines.tail(26)

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear
#Out[44]: 48.30985585336209
##################### Exponential ##############################

Exp = smf.ols('log_passengers ~ t', data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#Out[48]: 43.47847070446902
#################### Quadratic ###############################

Quad = smf.ols('Passengers ~ t + t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad
#Out[53]: 43.89814496742656
################### Additive seasonality ########################

add_sea = smf.ols('Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
#Out[90]: 124.97569951828822
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#Out[96]: 129.6291447652631
################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Passengers ~ t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#Out[102]: 30.39304289591171
################## Multiplicative Seasonality and Exponential ###########

Mul_Add_sea = smf.ols('log_passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
#Out[106]: 11.724791415163637
################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

# so rmse_Mult_add_sea has the least value among the models prepared so far 



