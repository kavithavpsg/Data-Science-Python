# Load the dataset
library(readxl)
crimedata <- read_csv("D:/Data science/Module 17 - K means clustering/crime_data (3)/crime_data (3).csv")

# Normalize the data
normalized_data <- scale(crimedata[, 2:5]) # excluding state name

d <- dist(normalized_data, method = "euclidean") # Distance matrix
d
fit <- hclust(d, method = "complete")

plot(fit) # Display dendrogram
plot(fit, hang = -1)

groups <- cutree(fit, k = 2) # Cut tree into 2 clusters

rect.hclust(fit, k = 2, border = "red")

membership <- as.matrix(groups)

final <- data.frame(membership, crimedata)

aggregate(crimedata[,2:5], by=list(final$membership), FUN = mean)

library(readr)
write_csv(final, "crimedataclustoutput_R.csv")

getwd()
