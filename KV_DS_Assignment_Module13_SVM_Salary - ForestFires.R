#####Support Vector Machines 
# Load the Dataset
Forest <- read.csv(file.choose())
# Partition Data into train and test data
Forest_train <- Forest[1:400, ]
Forest_test  <- Forest[401:517, ]

# Training a model on the data ----
# Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)
Forest_classifier <- ksvm(size_category ~ ., data = Forest_train, kernel = "vanilladot")

## Evaluating model performance ----
# predictions on testing dataset
Forest_predictions <- predict(Forest_classifier, Forest_test)

table(Forest_predictions, Forest_test$size_category)
agreement <- Forest_predictions == Forest_test$size_category
table(agreement)
prop.table(table(agreement))

