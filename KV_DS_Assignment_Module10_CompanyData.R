# Load the Data
# company_data.csv

##Exploring and preparing the data ----
company_data = read.csv("D:/Data science/Module 10 - DT and RF/Assignment/Fraud_check (1)/Company_Data (1).csv")

# look at the class variable
table(company_data$Sales)
mean(company_data$Sales)
fun1 <- function(x) if (x <= 7.49) {0} else {1}
company_data$SalesCategory <- mapply(fun1, company_data$Sales)
company_data$SalesCategory = as.factor(company_data$SalesCategory)
# Shuffle the data
company_data_rand <- company_data[order(runif(400)), ]
str(company_data_rand)


# split the data frames
company_data_train <- company_data_rand[1:300, ]
company_data_test  <- company_data_rand[301:400, ]

# check the proportion of class variable
prop.table(table(company_data_rand$SalesCategory))



# Step 3: Training a model on the data
install.packages("C50")
library(C50)

company_data_model <- C5.0(company_data_train, company_data_train$SalesCategory)
windows()
plot(company_data_model) 

# Display detailed information about the tree
summary(company_data_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(company_data_model, company_data_test)
test_acc <- mean(company_data_test$SalesCategory == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(company_data_test$SalesCategory, test_res, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(company_data_model, company_data_train)
train_acc <- mean(company_data_train$SalesCategory == train_res)
train_acc
