# -*- coding: utf-8 -*-
"""
Created on Sun May  3 07:17:03 2020

@author: KavithaV
"""
import pandas as pd
import numpy as np
##from scipy.stats import kurtosis
###Question1_a
carspeeddist = pd.read_csv("D:/Data science/Module 3 - GraphRep/Assignments_module03/Q1_a.csv")
carspeeddist = carspeeddist.drop(['Index'],axis=1)
carspeeddist.speed.skew()
carspeeddist.dist.skew()
carspeeddist.speed.kurt()
carspeeddist.dist.kurt()

###Question1_b
cartopspeed = pd.read_csv("D:/Data science/Module 3 - GraphRep/Assignments_module03/Q2_b.csv")
cartopspeed = cartopspeed.drop(['Unnamed: 0'],axis=1)
cartopspeed.SP.skew()
cartopspeed.WT.skew()
cartopspeed.SP.kurt()
cartopspeed.WT.kurt()

##Question2
from scipy.stats import t

# define probability and degrees of freedom
p = 0.96
# retrieve value <= probability 
value = t.ppf(p,1999)
print(value)
# confirm with cdf
p = t.cdf(value,1999)
print(p)

p = 0.94
# retrieve value <= probability
value = t.ppf(p,1999)
print(value)
# confirm with cdf
p = t.cdf(value,1999)
print(p)

p = 0.98
# retrieve value <= probability
value = t.ppf(p,1999)
print(value)
# confirm with cdf
p = t.cdf(value,1999)
print(p)

marks=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
print(np.mean(marks))
print(np.median(marks))
print(max(marks,key=marks.count))
print(np.std(marks))
print(np.var(marks))
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
stats.probplot(marks,dist="norm",plot=pylab)
