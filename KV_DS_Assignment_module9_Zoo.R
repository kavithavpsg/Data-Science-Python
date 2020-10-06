# Load the wbcd dataset
Zoo <- read.csv("D:/Data science/Module 9 - KNN/Assignment/Zoo (1)/Zoo (1).csv")

summary(Zoo)

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the Zoo data
Zoo_n <- as.data.frame(lapply(Zoo[1:18], normalize))

summary(Zoo_n)

# create training and test data
Zoo_train <- Zoo[1:75, ]
Zoo_test <- Zoo[76:101, ]

# create labels for training and test data
Zoo_train_labels <- Zoo[1:75, 10]
Zoo_test_labels <- Zoo[76:101, 10]

#---- Training a model on the data ----
# load the "class" library
install.packages("class")
library(class)

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k = 21)

##--------Evaluating model performance ----
confusion_test <- table(x = Zoo_test_labels, y = Zoo_test_pred)

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy 

# Training Accuracy to compare against test accuracy
Zoo_train_pred <- knn(train = Zoo_train, test = Zoo_train, cl = Zoo_train_labels, k=11)

confusion_train <- table(x = Zoo_train_labels, y = Zoo_train_pred)
Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train
