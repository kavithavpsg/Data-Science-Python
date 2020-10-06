############ Diabetes #################
# Read the dataset
Diabetes <- read.csv("D:/Data science/Module 10 - DT and RF/Assignment/Fraud_check (1)/Diabetes (1).csv")

#Create a function to normalize the data
norm <- function(x){ 
  return((x-min(x))/(max(x)-min(x)))
}

#Apply the normalization function to Diabetes dataset
Diabetes_n <- as.data.frame(lapply(Diabetes[1:8], norm))

# create training and test data
Diabetes_train <- Diabetes_n[1:500, ]
Diabetes_test <- Diabetes_n[501:768, ]

# create labels for training and test data
Diabetes_train_labels <- Diabetes[1:500, 9]
Diabetes_test_labels <- Diabetes[501:768, 9]


# Building a random forest model on training data 
install.packages("randomForest")
library(randomForest)

Diabetes_forest <- randomForest(Diabetes_train_labels ~ .,data=Diabetes_train,importance=TRUE)
plot(Diabetes_forest)

# Test Data Accuracy
test_acc <- mean(Diabetes_test_labels == predict(Diabetes_forest, newdata=Diabetes_test))
test_acc

# Train Data Accuracy
train_acc <- mean(Diabetes_train_labels == predict(Diabetes_forest, data=Diabetes_train))
train_acc
