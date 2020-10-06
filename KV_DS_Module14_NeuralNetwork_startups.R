# Load the startups data
startups <- read.csv("D:/Data science/Module 14 - NN/50_Startups (3)/50_Startups (3).csv")

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

#Encoding categorical column
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}
table(startups[["State"]], encode_ordinal(startups[["State"]]), useNA = "ifany")
startups = subset(startups, select = c(R.D.Spend,Administration,Marketing.Spend,Profit))
# Apply normalization function to entire data
startups_norm <- as.data.frame(lapply(startups, normalize))

# create training and test data
startups_train <- startups_norm[1:35, ]
startups_test <- startups_norm[35:50, ]

# Train the neuralnet model
install.packages("neuralnet")
library(neuralnet)

# ANN with only a single hidden neuron
startups_model <- neuralnet(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = startups_train)
# visualize the network topology
plot(startups_model)

## Evaluating model performance
model_results <- compute(startups_model, startups_test[1:4])

# Obtain predicted profit values
predicted_Profit <- model_results$net.result

# Examine the correlation between predicted and actual values
cor(predicted_Profit, startups_test$Profit)

## Improving model performance
# A more complex neural network topology with 5 hidden neurons
startups_model2 <- neuralnet(Profit ~ R.D.Spend + Administration + Marketing.Spend,data = startups_train, hidden = 5)

# plot the network
plot(startups_model2)

# Evaluate the results
model_results2 <- compute(startups_model2, startups_test[1:4])
predicted_Profit2 <- model_results2$net.result
cor(predicted_Profit2, startups_test$Profit)
