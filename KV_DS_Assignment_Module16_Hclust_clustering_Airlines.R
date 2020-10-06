# Load the dataset
library(readxl)
airlines <- read_excel("D:/Data science/Module 16 - Hierarichal clustering/crime_data (2)/EastWestAirlines (3).xlsx",sheet='data')

# Normalize the data
normalized_data <- scale(airlines[, 2:12]) # Excluding the university name

d <- dist(normalized_data, method = "euclidean") # Distance matrix
d
fit <- hclust(d, method = "complete")

plot(fit) # Display dendrogram
plot(fit, hang = -1)

groups <- cutree(fit, k = 5) # Cut tree into 5 clusters

rect.hclust(fit, k = 5, border = "red")

membership <- as.matrix(groups)

final <- data.frame(membership, airlines)

aggregate(airlines[,2:7], by=list(final$membership), FUN = mean)

library(readr)
write_csv(final, "airlinesclustoutput.csv")

getwd()
