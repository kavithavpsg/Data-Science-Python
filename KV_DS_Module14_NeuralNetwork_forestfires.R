# Load the forestfires data
forestfires <- read.csv("D:/Data science/Module 14 - NN/50_Startups (3)/forestfires (2).csv")
glimpse(forestfires)
# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

#Encoding categorical column
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}
table(forestfires[["size_category"]], encode_ordinal(forestfires[["size_category"]]), useNA = "ifany")
table(forestfires[["month"]], encode_ordinal(forestfires[["month"]]), useNA = "ifany")
table(forestfires[["day"]], encode_ordinal(forestfires[["day"]]), useNA = "ifany")
#forestfires = subset(forestfires, select = c(R.D.Spend,Administration,Marketing.Spend,Profit))
# Apply normalization function to numerical data
forestfires_norm <- cbind(lapply(forestfires[3:10], normalize),forestfires[11:31])
forestfires_norm$size_category <- ifelse(forestfires_norm$size_category == "small",1,0)

# create training and test data
forestfires_train <- forestfires_norm[1:400, ]
forestfires_test <- forestfires_norm[401:517, ]

# Train the neuralnet model
install.packages("neuralnet")
library(neuralnet)
colnames(forestfires_norm)

#
# ANN with only a single hidden neuron
forestfires_model <- neuralnet(formula = size_category ~ FFMC+DMC+DC+ISI+temp+RH+wind+rain+area+dayfri+daymon+daysat+daysun+daythu+daytue+daywed+monthapr+monthaug+monthdec+monthfeb+monthjan+monthjul+monthjun+monthmar+monthmay+monthnov+monthoct+monthsep, data = forestfires_train)
# visualize the network topology
plot(forestfires_model)

## Evaluating model performance
model_results <- neuralnet::compute(forestfires_model, forestfires_test[1:28])

# Obtain predicted profit values
predicted_sizecategory <- model_results$net.result

# Examine the correlation between predicted and actual values
cor(predicted_sizecategory, forestfires_test$size_category)

## Improving model performance
# A more complex neural network topology with 5 hidden neurons
forestfires_model2 <- neuralnet(size_category ~ FFMC+DMC+DC+ISI+temp+RH+wind+rain+area+dayfri+daymon+daysat+daysun+daythu+daytue+daywed+monthapr+monthaug+monthdec+monthfeb+monthjan+monthjul+monthjun+monthmar+monthmay+monthnov+monthoct+monthsep,data = forestfires_train, hidden = 5)

# plot the network
plot(forestfires_model2)

# Evaluate the results
model_results2 <- neuralnet::compute(forestfires_model2, forestfires_test[1:28])
predicted_sizecategory2 <- model_results2$net.result
cor(predicted_sizecategory2, forestfires_test$sizecategory)
