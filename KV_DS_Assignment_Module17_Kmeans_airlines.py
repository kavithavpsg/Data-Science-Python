# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:01:12 2020

@author: gauth
"""

import pandas as pd
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

airlines= pd.read_excel("D:/Data science/Module 17 - K means clustering/crime_data (3)/EastWestAirlines (4).xlsx",sheet_name = 'data')

airlines.describe()


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
##Check for null values
airlines.isnull().sum()

##Check for outliers
columns = ['Balance','Qual_miles','cc1_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Days_since_enroll','Award?']
for value in columns:
    plt.figure()
    plt.boxplot(airlines[value])
# Normalized data frame (considering the numerical part of data)
airlines_norm = norm_func(airlines.iloc[:,1:])
airlines_norm.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    TWSS.append(kmeans.inertia_)

TWSS
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters=4)
model.fit(airlines_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines['clust'] = mb # creating a  new column and assigning it to new column 

airlines.groupby(airlines.clust).mean()

# creating a csv file 
airlines.to_csv("Kmeans_kmeans.csv", encoding = "utf-8")

