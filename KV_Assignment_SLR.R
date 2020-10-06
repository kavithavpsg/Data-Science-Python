calories <- read.csv("D:/Data science/Module 6 - LR/Assignment/calories_consumed.csv")
attach(calories)
summary(calories)
colnames(calories) <- c("Weight","Cal")
attach(calories)
plot(Cal,Weight,) # scatter plot
cor(Cal, Weight) # correlation coefficient

model <- lm(Weight ~ Cal) # linear regression
model
summary(model) # output and evaluating

train<-calories[1:10, ]
test<-calories[11:14, ]

predict(model,data=train)
result<-predict(model,data=test)
result

model$residuals

confint(model,level=0.95)
predict(model, interval = "confidence")

rmse <- sqrt(mean(model$residuals^2))
rmse


# Log model
plot(log(Cal), Weight)
cor(log(Cal), Weight)
model2 <- lm(Weight ~ log(Cal)) # log transformation
summary(model2)

rmse2 <- sqrt(mean(model2$residuals^2))
rmse2


# Exp model
plot(Cal, Weight)
cor(Cal, log(Weight))
model3 <- lm(log(Weight) ~ Cal) # exponential tranformation
summary(model3)
model3$residuals

log_Wt <- predict(model3,interval="confidence")
log_Wt
WEIGHT <- exp(log_Wt)
WEIGHT

err <- Weight-WEIGHT
err

rmse3 <- sqrt(mean(err^2))
rmse3


# Polynomial transformation
model4 <- lm(log(Weight) ~ Cal + I(Cal * Cal))
summary(model4)

confint(model4,level=0.95)

log_res <- predict(model4,interval="confidence")
Wtpoly <- exp(log_res)
Wtpoly
err_poly <- Weight - Wtpoly
err_poly

rmse4 <- sqrt(mean(err_poly^2))
rmse4

##Build a prediction model for Churn_out_rate 
Emp <- read.csv("D:/Data science/Module 6 - LR/Assignment/emp_data.csv")
attach(Emp)
summary(Emp)
plot(Salary_hike,Churn_out_rate) # scatter plot
cor(Salary_hike, Churn_out_rate) # correlation coefficient
 
model <- lm(Churn_out_rate ~ Salary_hike) # linear regression
model
summary(model) # output and evaluating

train<-delivery[1:7, ]
test<-delivery[7:12, ]

predict(model,data=train)
result<-predict(model,data=test)
result

model$residuals

confint(model,level=0.95)
predict(model, interval = "confidence")

rmse <- sqrt(mean(model$residuals^2))
rmse


# Log model
plot(log(Salary_hike), Churn_out_rate)
cor(log(Salary_hike), Churn_out_rate)
model2 <- lm(Salary_hike ~ log(Churn_out_rate)) # log transformation
summary(model2)

rmse2 <- sqrt(mean(model2$residuals^2))
rmse2


# Exp model
plot(Salary_hike, Churn_out_rate)
cor(Salary_hike, log(Churn_out_rate))
model3 <- lm(log(Churn_out_rate) ~ Salary_hike) # exponential tranformation
summary(model3)
model3$residuals

log_Churn <- predict(model3,interval="confidence")
log_Churn
logChurn <- exp(log_Churn)
logChurn

err <- Churn_out_rate-logChurn
err

rmse3 <- sqrt(mean(err^2))
rmse3


# Polynomial transformation
model4 <- lm(log(Churn_out_rate) ~ Salary_hike + I(Churn_out_rate * Churn_out_rate))
summary(model4)

confint(model4,level=0.95)

log_res <- predict(model4,interval="confidence")
Churnpoly <- exp(log_res)
Churnpoly
err_poly <- Churn_out_rate - Churnpoly
err_poly

rmse4 <- sqrt(mean(err_poly^2))
rmse4

##Build a prediction model for Salary_hike
salary <- read.csv("D:/Data science/Module 6 - LR/Assignment/Salary_Data.csv")
attach(salary)
summary(Emp)
plot(YearsExperience,Salary) # scatter plot
cor(YearsExperience, Salary) # correlation coefficient

model <- lm(Salary ~ YearsExperience) # linear regression
model
summary(model) # output and evaluating

train<-delivery[1:7, ]
test<-delivery[7:12, ]

predict(model,data=train)
result<-predict(model,data=test)
result

model$residuals

confint(model,level=0.95)
predict(model, interval = "confidence")

rmse <- sqrt(mean(model$residuals^2))
rmse


# Log model
plot(log(YearsExperience), Salary)
cor(log(YearsExperience), Salary)
model2 <- lm(YearsExperience ~ log(Salary)) # log transformation
summary(model2)

rmse2 <- sqrt(mean(model2$residuals^2))
rmse2


# Exp model
plot(YearsExperience, Salary)
cor(YearsExperience, log(Salary))
model3 <- lm(log(Salary) ~ YearsExperience) # exponential tranformation
summary(model3)
model3$residuals

log_Salary <- predict(model3,interval="confidence")
log_Salary
logSalary <- exp(log_Salary)
logSalary

err <- Salary-logSalary
err

rmse3 <- sqrt(mean(err^2))
rmse3


# Polynomial transformation
model4 <- lm(log(Salary) ~ YearsExperience + I(Salary * Salary))
summary(model4)

confint(model4,level=0.95)

log_res <- predict(model4,interval="confidence")
Salarypoly <- exp(log_res)
Salarypoly
err_poly <- Salary - Salarypoly
err_poly

rmse4 <- sqrt(mean(err_poly^2))
rmse4

