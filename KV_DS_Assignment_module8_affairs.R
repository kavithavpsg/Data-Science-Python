# Load the Dataset

Affairs <- read.csv(file.choose()) # Choose the Affairs Data set

colnames(Affairs)
Affairs <- Affairs[, -1] # Removing the first column which is is an Index
fun1 <- function(x) if (x != 0) {1} else {0}
Affairs$naffairs <- mapply(fun1, Affairs$naffairs)
Affairs

Affairs_train <- Affairs[1:500, ]
Affairs_test <- Affairs[501:601, ]

model <- glm(naffairs~., data = Affairs_train, family = "binomial")
summary(model)

# Prediction on Test data 
prob_test <- predict(model, Affairs_test, type="response")
prob_test

# Confusion matrix and considering the threshold value as 0.5 
confusion_test <- table(prob_test>0.5, Affairs_test$ATTORNEY)
confusion_test

# Model Accuracy 
sum(diag(confusion_test))
Accuracy_test <- sum(diag(confusion_test)) / sum(confusion_test)
Accuracy_test


# Prediction on Train data 
prob_train <- predict(model, Affairs_train, type="response")

# Confusion matrix and considering the threshold value as 0.5 
confusion_train <- table(prob_train > 0.5, Affairs_train$naffairs)
confusion_train

# Model Accuracy 
Accuracy_train <- sum(diag(confusion_train)) / sum(confusion_train)
Accuracy_train

