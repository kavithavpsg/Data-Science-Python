# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 07:30:40 2020

@author: KavithaV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

#####Question 7
##Read data
assignment_2 = pd.read_excel("D:\Data science\Module 2 - EDA\Hands On_Module 2\Assignment_module02.xlsx")

##Calculate first moment
assignment_2.Points.mean()
assignment_2.Score.mean()
assignment_2.Weigh.mean()


assignment_2.Points.median()
assignment_2.Score.median()
assignment_2.Weigh.median()


assignment_2.Points.mode()
assignment_2.Score.mode()
assignment_2.Weigh.mode()

##Calculate second moment
assignment_2.Points.var()
assignment_2.Score.var()
assignment_2.Weigh.var()

assignment_2.Points.std()
assignment_2.Score.std()
assignment_2.Weigh.std()

pointsrange = max(assignment_2.Points)-min(assignment_2.Points)
pointsrange

scorerange = max(assignment_2.Score) - min(assignment_2.Score)
scorerange

weighrange = max(assignment_2.Weigh) - min(assignment_2.Weigh)
weighrange


##Calculate third moment
assignment_2.Points.skew()
assignment_2.Score.skew()
assignment_2.Weigh.skew()

assignment_2.Points.kurt()
assignment_2.Score.kurt()
assignment_2.Weigh.kurt()

##calculate correlation coefficient
cov1 = np.cov(assignment_2.Points,assignment_2.Weigh)
##calculate pearson correlation coefficient
corr1,_= sc.pearsonr(assignment_2.Points,assignment_2.Weigh)
corr1

cov2 = np.cov(assignment_2.Score,assignment_2.Weigh)
cov2
corr2,_= sc.pearsonr(assignment_2.Score,assignment_2.Weigh)
corr2


####Question 8
wt_of_patients = [308, 330, 323, 334, 335, 345, 367, 387, 399]
round(np.mean(wt_of_patients),2)

###Question 9
company=pd.read_csv("D:/Data science/Module 2 - EDA/Company.csv")
plt.scatter(company['Name of company'],company['Measure X'])
plt.xticks(rotation=90)

mean_company = np.mean(company)
mean_company
stddev_company = np.std(company)
stddev_company
var_company = np.var(company)
var_company

