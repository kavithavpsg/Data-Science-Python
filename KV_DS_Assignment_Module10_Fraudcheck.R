# Load the Data
Fraud_check = read.csv("D:/Data science/Module 10 - DT and RF/Assignment/Fraud_check (1)/Fraud_check (1).csv")

##Exploring and preparing the data ----
str(Fraud_check)

# look at the class variable
table(Fraud_check)
fun1 <- function(x) if (x <= 30000) {'Risky'} else {'Good'}
Fraud_check$RiskCategory <- mapply(fun1, Fraud_check$Taxable.Income)

# Shuffle the data
Fraud_rand <- Fraud_check[order(runif(600)), ]
str(Fraud_rand)

Fraud_rand$RiskCategory <- as.factor(Fraud_rand$RiskCategory)
df1_complete <- na.omit(df1) # Method 1 - Remove NA
df1_complete
# Fraud_train$Undergrad <- as.factor(Fraud_train$Undergrad)
# Fraud_train$Marital.Status <- as.factor(Fraud_train$Marital.Status)
# Fraud_train$Urban <- as.factor(Fraud_train$Urban)

# split the data frames
Fraud_train <- Fraud_rand[1:500, ]
Fraud_test  <- Fraud_rand[501:600, ]

# check the proportion of class variable
prop.table(table(Fraud_rand$RiskCategory))



# Step 3: Training a model on the data
install.packages("C50")
library(C50)

Fraud_model <- C5.0(Fraud_train[,-6], Fraud_train$RiskCategory)
windows()
plot(Fraud_model) 

# Display detailed information about the tree
summary(Fraud_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(Fraud_model, Fraud_test)
test_acc <- mean(Fraud_test$RiskCategory == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(Fraud_test$RiskCategory, test_res, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual RiskCategory', 'predicted RiskCategory'))

# On Training Dataset
train_res <- predict(Fraud_model, Fraud_train)
train_acc <- mean(Fraud_train$RiskCategory == train_res)
train_acc
