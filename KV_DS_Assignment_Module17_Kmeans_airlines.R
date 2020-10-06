library(readxl)
airlines <- read_excel("D:/Data science/Module 17 - K means clustering/crime_data (3)/EastWestAirlines (4).xlsx",sheet = 'data')

# Normalize the data
normalized_data <- scale(airlines[, 2:12]) # 

# Elbow curve & k ~ sqrt(n/2) to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(normalized_data, centers=i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

fit <- kmeans(normalized_data, 4) # 4 cluster solution
str(fit)
fit$cluster
final <- data.frame(fit$cluster, airlines) # Append cluster membership

aggregate(airlines[, 2:7], by=list(fit$cluster), FUN = mean)
