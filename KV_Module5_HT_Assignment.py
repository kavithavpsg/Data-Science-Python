# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:08:10 2020

@author: KavithaV
"""
import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm
##Qn 1 - cutlet diamter
cutlet = pd.read_csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Cutlets.csv")
cutlet.describe()
cutlet.columns = "UnitA","UnitB"
# Normality Test
print(stats.shapiro(cutlet.UnitA)) # Shapiro Test
print(stats.shapiro(cutlet.UnitB))

# Variance test
scipy.stats.levene(cutlet.UnitA, cutlet.UnitB)

# p-value = 0.4176 > 0.05 so p high null fly => Equal variances

# 2 Sample T test
scipy.stats.ttest_ind(cutlet.UnitA, cutlet.UnitB)


#Qn 2 - average TAT of 4 different labs
############# One - Way Anova ################
from statsmodels.formula.api import ols

labTAT=pd.read_csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/LabTAT.csv")
labTAT
labTAT.columns="LabA","LabB","LabC","LabD"

# Normality Test
print(stats.shapiro(labTAT.LabA)) # Shapiro Test
print(stats.shapiro(labTAT.LabB))
print(stats.shapiro(labTAT.LabC))
print(stats.shapiro(labTAT.LabD))

# Variance test
scipy.stats.levene(labTAT.LabA, labTAT.LabB)
scipy.stats.levene(labTAT.LabB, labTAT.LabC)
scipy.stats.levene(labTAT.LabC, labTAT.LabD)
scipy.stats.levene(labTAT.LabB, labTAT.LabD)
scipy.stats.levene(labTAT.LabC, labTAT.LabA)
scipy.stats.levene(labTAT.LabD, labTAT.LabA)

# One - Way Anova
mod = ols('LabA ~ LabB + LabC + LabD',data = labTAT).fit()

aov_table = sm.stats.anova_lm(mod, type=2)
help(sm.stats.anova_lm)

aov_table

######### 2-proportion test ###########
import numpy as np

Male_Female_Prop_test = pd.read_csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Buyer Ratio.csv")

from statsmodels.stats.proportion import proportions_ztest

nobs = np.array([393,1731])

count_East = np.array([50,550])

stats, pval = proportions_ztest(count_East, nobs, alternative='two-sided') 
print(pval) # Pvalue- 0.000  
stats, pval = proportions_ztest(count_East, nobs, alternative='larger')
print(pval)  # Pvalue 0.999  

count_West = np.array([142,351])

stats, pval = proportions_ztest(count_West, nobs, alternative='two-sided') 
print(pval) # Pvalue- 0.000  
stats, pval = proportions_ztest(count_West, nobs, alternative='larger')
print(pval)  # Pvalue 0.999  
stats, pval = proportions_ztest(count_West, nobs, alternative='smaller')
print(pval)  # Pvalue 0.999  
count_North = np.array([131,480])

stats, pval = proportions_ztest(count_North, nobs, alternative='two-sided') 
print(pval) # Pvalue<0.05 
stats, pval = proportions_ztest(count_North, nobs, alternative='larger')
print(pval)  # Pvalue<0.05  
stats, pval = proportions_ztest(count_North, nobs, alternative='smaller')
print(pval)  # Pvalue 0.999 
 
count_South = np.array([70,350])

stats, pval = proportions_ztest(count_South, nobs, alternative='two-sided') 
print(pval) # Pvalue- 0.279  
stats, pval = proportions_ztest(count_South, nobs, alternative='larger')
print(pval)  # Pvalue 0.8603  
stats, pval = proportions_ztest(count_South, nobs, alternative='smaller')
print(pval)  # Pvalue 0.139  

################ Chi-Square Test ################

customer_order=pd.read_csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Customer OrderForm.csv")
cust_order_def = []
cust_order_country = []
for rindex in range(300):
    for cindex in range(4):
        if cindex == 0:
            cust_order_def += [customer_order.iloc[rindex][cindex]]
            cust_order_country += ['Phillippines']
        elif cindex == 1:
            cust_order_def += [customer_order.iloc[rindex][cindex]]
            cust_order_country += ['Indonesia']
        elif cindex == 2:
            cust_order_def += [customer_order.iloc[rindex][cindex]]
            cust_order_country += ['Malta']
        else:
            cust_order_def += [customer_order.iloc[rindex][cindex]]
            cust_order_country += ['India']
list_of_tuples = list(zip(cust_order_def,cust_order_country))
Cust_order = pd.DataFrame(list_of_tuples, columns = ['Defective', 'Country'])  
count=pd.crosstab(Cust_order["Defective"], Cust_order["Country"])
count
Chisquares_results=scipy.stats.chi2_contingency(count)

Chi_square=[['', 'Test Statistic', 'p-value'],['Sample Data', Chisquares_results[0], Chisquares_results[1]]]
Chi_square

Fantaloons = pd.read_csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Faltoons.csv")
count_Fantaloons = pd.crosstab(Fantaloons["Weekdays"], Fantaloons["Weekend"])
count_Fantaloons
Chisquares_results_Fantaloons = scipy.stats.chi2_contingency(count_Fantaloons)

Chi_square_Fantaloons = [['', 'Test Statistic', 'p-value'],['Sample Data', Chisquares_results_Fantaloons[0], Chisquares_results_Fantaloons[1]]]
Chi_square_Fantaloons
