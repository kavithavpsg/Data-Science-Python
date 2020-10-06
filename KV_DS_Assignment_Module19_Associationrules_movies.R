install.packages("arules")
install.packages("arulesViz")

library("arules")
library(arulesViz)

movies = read.csv("D:/Data science/Module 19 - Association/Data Sets (1)/my_movies.csv")
summary(movies)
inspect(movies[1:5])

rules <- apriori(movies, parameter = list(support = 0.005, confidence = 0.6, minlen = 2))
inspect(rules[1:5])

windows()
plot(rules, method = "scatterplot")
plot(rules, method = "grouped")

rules <- sort(rules,by="lift")

inspect(rules[1:5])
