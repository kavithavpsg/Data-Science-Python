# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:12:46 2020

@author: KavithaV
"""
import pandas as pd
import matplotlib.pylab as plt

crimedata= pd.read_csv("D:/Data science/Module 16 - Hierarichal clustering/crime_data (2)/crime_data (2).csv")

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

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(crimedata_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=2, linkage = 'complete', affinity = "euclidean").fit(crimedata_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

crimedata['clust']=cluster_labels # creating a new column and assigning it to new column 

# Aggregate mean of each cluster
crimedata.groupby(crimedata.clust).mean()

# creating a csv file 
crimedata.to_csv("crimedata.csv", encoding = "utf-8")
