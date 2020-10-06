# Load the wbcd dataset
glass <- read.csv("D:/Data science/Module 9 - KNN/Assignment/Zoo (1)/glass (1).csv")

summary(glass)

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the glass data
glass_n <- as.data.frame(lapply(glass[1:10], normalize))

summary(glass_n)

# create training and test data
glass_train <- glass_n[1:50, ]
glass_test <- glass_n[151:214, ]

# create labels for training and test data
glass_train_labels <- glass[1:150, 10]
glass_test_labels <- glass[151:214, 10]

#---- Training a model on the data ----
# load the "class" library
install.packages("class")
library(class)

glass_test_pred <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k = 21)

##--------Evaluating model performance ----
confusion_test <- table(x = glass_test_labels, y = glass_test_pred)

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy 

# Training Accuracy to compare against test accuracy
glass_train_pred <- knn(train = glass_train, test = glass_train, cl = glass_train_labels, k=11)

confusion_train <- table(x = glass_train_labels, y = glass_train_pred)
Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train
