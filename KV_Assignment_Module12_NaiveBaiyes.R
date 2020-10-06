# Import the raw_salary dataset
library(readr)
Salary_train <- read_csv("D:/Data science/Module 12-Bayesian/Assignments/Bayesian Classifier_Assignments/SalaryData_Train.csv")
Salary_test <- read_csv("D:/Data science/Module 12-Bayesian/Assignments/Bayesian Classifier_Assignments/SalaryData_Test.csv")
View(Salary_train)

str(Salary_train)

Salary_train$Salary <- factor(Salary_train$Salary)
Salary_train$workclass <- factor(Salary_train$workclass)
Salary_train$education <- factor(Salary_train$education)
Salary_train$maritalstatus <- factor(Salary_train$maritalstatus)
Salary_train$occupation <- factor(Salary_train$occupation)
Salary_train$relationship <- factor(Salary_train$relationship)
Salary_train$race <- factor(Salary_train$race)
Salary_train$sex <- factor(Salary_train$sex)
Salary_train$native <- factor(Salary_train$native)


Salary_test$Salary <- factor(Salary_test$Salary)
Salary_test$workclass <- factor(Salary_test$workclass)
Salary_test$education <- factor(Salary_test$education)
Salary_test$maritalstatus <- factor(Salary_test$maritalstatus)
Salary_test$occupation <- factor(Salary_test$occupation)
Salary_test$relationship <- factor(Salary_test$relationship)
Salary_test$race <- factor(Salary_test$race)
Salary_test$sex <- factor(Salary_test$sex)
Salary_test$native <- factor(Salary_test$native)


# proportion of ham and spam messages
prop.table(table(Salary_train$Salary))
prop.table(table(Salary_test$Salary))
library(e1071)
Naive_Bayes_Model=naiveBayes(Salary ~., data=Salary_train)

Naive_Bayes_Model

#Prediction on the dataset
NB_Predictions=predict(Naive_Bayes_Model,Salary_test)
#Confusion matrix to check accuracy
table(NB_Predictions,Salary_test$Salary)


test_acc = mean(NB_Predictions == Salary_test$Salary)
test_acc

library(gmodels)
CrossTable(NB_Predictions, Salary_test$Salary, dnn = c('predicted', 'actual'))

salary_train_pred <- predict(Naive_Bayes_Model, Salary_train)
train_acc = mean(salary_train_pred == Salary_train$Salary)
train_acc
