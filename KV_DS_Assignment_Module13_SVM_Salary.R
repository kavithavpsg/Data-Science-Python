#####Support Vector Machines 
# Load the Dataset
Salary_train <- read.csv(file.choose())
Salary_test <- read.csv(file.choose())

# Training a model on the data ----
# Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)
Salary_classifier <- ksvm(Salary ~ ., data = Salary_train, kernel = "vanilladot")

## Evaluating model performance ----
# predictions on testing dataset
Salary_predictions <- predict(Salary_classifier, Salary_test)

table(Salary_predictions, Salary_test$Salary)
agreement <- Salary_predictions == Salary_test$Salary
table(agreement)
prop.table(table(agreement))

## Improving model performance ----
Salary_classifier_rbf <- ksvm(Salary ~ ., data = Salary_train, kernel = "rbfdot")
Salary_predictions_rbf <- predict(Salary_classifier_rbf, Salary_test)
agreement_rbf <- Salary_predictions_rbf == Salary_test$Salary
table(agreement_rbf)
prop.table(table(agreement_rbf))
