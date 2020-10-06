# Load the Dataset

Bank <- read.csv(file.choose()) # Choose the Bank Data set

colnames(Bank)
Bank <- Bank[, -8,-9,-10,-11,-13,-14,-15] # Removing the first column which is is an Index

Bank_train <- Bank[1:35000, ]
Bank_test <- Bank[35001:45211, ]

model <- glm(y~., data = Bank_train, family = "binomial")
summary(model)

# Prediction on Test data 
prob_test <- predict(model, Bank_test, type="response")
prob_test

# Confusion matrix and considering the threshold value as 0.5 
confusion_test <- table(prob_test>0.5, Bank_test$y)
confusion_test

# Model Accuracy 
sum(diag(confusion_test))
Accuracy_test <- sum(diag(confusion_test)) / sum(confusion_test)
Accuracy_test


# Prediction on Train data 
prob_train <- predict(model, Bank_train, type="response")

# Confusion matrix and considering the threshold value as 0.5 
confusion_train <- table(prob_train > 0.5, Bank_train$ATTORNEY)
confusion_train

# Model Accuracy 
Accuracy_train <- sum(diag(confusion_train)) / sum(confusion_train)
Accuracy_train

#rocr curve
install.packages("ROCR")
library(ROCR)
rocr_pred <- prediction(prob, train_model$y)
rocr_perf <- performance(rocr_pred, 'tpr','fpr') 

str(rocr_perf)


plot(rocr_perf,colorize=T,text.adj=c(-0.2,1.7))

rocr_cutoff <- data.frame(cut_off = rocr_perf@alpha.values[[1]],fpr=rocr_perf@x.values,tpr=rocr_perf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

rocr_cutoff$cut_off

library(dplyr)
rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off, 6)
rocr_cutoff <- arrange(rocr_cutoff, desc(TPR))
View(rocr_cutoff$cut_off)
