# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 05:47:20 2020

@author: KavithaV
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

wine = pd.read_csv("D:/Data science/Module 18 - Dimensionality reduction/wine/wine.csv")
wine.describe()

wine.isnull().sum()

fig = px.scatter_matrix(wine)
fig.show()
##Find outliers
columns = ['Type','Alcohol','Malic','Ash','Alcalinity','Magnesium','Phenols','Flavanoids','Nonflavanoids','Proanthocyanins','Color','Hue','Dilution','Proline']
for value in columns:
    plt.figure()
    plt.boxplot(wine[value])

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 


# Normalizing the numerical data 
wine_normal = scale(wine)


pca = PCA(n_components = 5)
pca_wine = pca.fit_transform(wine_normal)

principal_wine_Df = pd.DataFrame(data = pca_wine, columns = ['pc1', 'pc2','pc3','pc4','pc5'])

# The amount of variance that each PCA explains is 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
##Pc 1 holds 39% of data, pc 2 holds 17% of data
var = pca.explained_variance_ratio_

plt.title("Principal Component Analysis of Wine Dataset",fontsize=20)
plt.scatter(principal_wine_Df['pc1'], principal_wine_Df['pc2'], alpha=0.8)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

pca.components_
pca.components_[0]
# Cumulative variance 

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

pca_wine
principal_wine_Df

###Perform Hierarchical clustering
#wine_norm = norm_func(principal_wine_Df.iloc[:,1:])
principal_wine_Df.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(principal_wine_Df, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=3, linkage = 'complete', affinity = "euclidean").fit(principal_wine_Df) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

principal_wine_Df['clust']=cluster_labels # creating a new column and assigning it to new column 

# Aggregate mean of each cluster
principal_wine_Df.groupby(principal_wine_Df.clust).mean()

# creating a csv file 
principal_wine_Df.to_csv("principal_wine_Df.csv", encoding = "utf-8")

from sklearn.cluster import	KMeans

TWSS = []
k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(principal_wine_Df)
    TWSS.append(kmeans.inertia_)

TWSS
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters=3)
model.fit(principal_wine_Df)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
principal_wine_Df['clust'] = mb # creating a  new column and assigning it to new column 

principal_wine_Df.groupby(principal_wine_Df.clust).mean()

# creating a csv file 
principal_wine_Df.to_csv("Kmeans_principal_wine.csv", encoding = "utf-8")



