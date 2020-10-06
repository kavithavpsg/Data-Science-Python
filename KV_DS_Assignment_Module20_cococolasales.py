# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:30:09 2020

@author: KavithaV
"""

import pandas as pd
cococola = pd.read_excel("D:/Data science/Module 20 - Forecast and Time series/CocaCola_Sales_Rawdata (1)/CocaCola_Sales_Rawdata (1).xlsx")
import numpy as np
# Data Preparation
cococola["t"] = np.arange(1,43)

cococola["t_squared"] = cococola["t"]*cococola["t"]
cococola.columns
cococola["log_sales"] = np.log(cococola["Sales"])
#Exract quarter from quarter/year column
cococola["Quart"] = cococola["Quarter"].str[0:2]
# Dummy variables for Seasons
months =['Q1','Q2','Q3','Q4'] 

month_dummies = pd.DataFrame(pd.get_dummies(cococola['Quart']))
cococola = pd.concat([cococola,month_dummies],axis = 1)

# Partition the data
cococola.iloc[:, 1].plot() # Timeplot
Train = cococola.head(30)
Test = cococola.tail(12)

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
#Out[21]: 714.0144483281297
##################### Exponential ##############################

Exp = smf.ols('log_sales ~ t', data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#Out[25]: 552.2821039079295
#################### Quadratic ###############################

Quad = smf.ols('Sales ~ t + t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
#Out[29]: 646.2715428311323
################### Additive seasonality ########################

add_sea = smf.ols('Sales ~ Q1+Q2+Q3+Q4', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
#Out[34]: 1778.0065467941363
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales ~ Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#Out[38]: 1828.9238912138073
################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Sales ~ t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#Out[42]: 586.0533067871024
################## Multiplicative Seasonality and Exponential ###########

Mul_Add_sea = smf.ols('log_sales ~ t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
#Out[47]: 410.24970596078435
################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

# so rmse_Mult_add_sea has the least value among the models prepared so far 



