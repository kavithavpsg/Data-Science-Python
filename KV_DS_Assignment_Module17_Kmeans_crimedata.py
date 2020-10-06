# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:50:31 2020

@author: KavithaV
"""

import pandas as pd
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

crimedata= pd.read_csv("D:/Data science/Module 17 - K means clustering/crime_data (3)/crime_data (3).csv")

crimedata.describe()


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
##Check for null values
crimedata.isnull().sum()
##Check for outliers
columns = ['Murder','Assault','UrbanPop','Rape']
for value in columns:
    plt.figure()
    plt.boxplot(crimedata[value])
# Normalized data frame (considering the numerical part of data)
crimedata_norm = norm_func(crimedata.iloc[:,1:])
crimedata_norm.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crimedata_norm)
    TWSS.append(kmeans.inertia_)

TWSS
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters=4)
model.fit(crimedata_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crimedata['clust'] = mb # creating a  new column and assigning it to new column 

crimedata.groupby(crimedata.clust).mean()

# creating a csv file 
crimedata.to_csv("Kmeans_crimedata.csv", encoding = "utf-8")

