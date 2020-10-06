library(readr)
crimedata <- read_csv("D:/Data science/Module 17 - K means clustering/crime_data (3)/crime_data (3).csv")

# Normalize the data
normalized_data <- scale(crimedata[, 2:5]) # 

# Elbow curve & k ~ sqrt(n/2) to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(normalized_data, centers=i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

fit <- kmeans(normalized_data, 4) # 5 cluster solution
str(fit)
fit$cluster
final <- data.frame(fit$cluster, crimedata) # Append cluster membership

aggregate(crimedata[, 2:5], by=list(fit$cluster), FUN = mean)
