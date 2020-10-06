# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:31:48 2020

@author: KavithaV
"""
import pandas as pd
import matplotlib.pylab as plt

airlines= pd.read_excel("D:/Data science/Module 16 - Hierarichal clustering/crime_data (2)/EastWestAirlines (3).xlsx",sheet_name='data')

airlines.describe()
airlines = airlines.drop(["ID#"],axis=1)

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
df_norm = norm_func(airlines.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=5, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

airlines['clust']=cluster_labels # creating a new column and assigning it to new column 

# Aggregate mean of each cluster
airlines.groupby(airlines.clust).mean()

# creating a csv file 
airlines.to_csv("airlines.csv", encoding = "utf-8")