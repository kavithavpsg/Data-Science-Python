# Loading Wine data
library(readr)
wine <- read_csv(file.choose())

## Categorical data removed from the dataset
wine <- input[, c(2:14)]
summary(wine)

wine_scale <- scale(wine)
?princomp
pcaObj <- princomp(wine_scale, cor = TRUE, scores = TRUE, covmat = NULL)

summary(pca_wine)

loadings(pca_wine)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

# All the PCA scores
pcaObj$scores
#Store first 3 pca scores
pca_wine <- pcaObj$scores[,1:3]
summary(pca_wine)
###Hierarichal clustering
d <- dist(pca_wine, method = "euclidean") # Distance matrix
d
fit <- hclust(d, method = "complete")

plot(fit) # Display dendrogram
plot(fit, hang = -1)

groups <- cutree(fit, k = 3) # Cut tree into 2 clusters

rect.hclust(fit, k = 3, border = "red")

membership <- as.matrix(groups)

final <- data.frame(membership, pca_wine)

aggregate(pca_wine, by=list(final$membership), FUN = mean)

##Kmeans clustering

# Elbow curve & k ~ sqrt(n/2) to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(pca_wine, centers=i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

fit <- kmeans(pca_wine, 3) # 3 cluster solution
str(fit)
fit$cluster
final <- data.frame(fit$cluster, pca_wine) # Append cluster membership

aggregate(pca_wine, by=list(fit$cluster), FUN = mean)

