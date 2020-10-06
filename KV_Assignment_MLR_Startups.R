startups <- read.csv("D:/Data science/Module 7-MLR/Assignment/50_Startups (2)/50_Startups (2).csv")
attach(startups)
summary(startups)
install.packages("Hmisc")
library(Hmisc)
#Scatter diagram

pairs(startups)
startups_new = subset(startups, select = -c(State))
#Correlation coefficient Matrix - Strength & Direction of correlation
cor(startups_new)

# The Linear Model of interest
model.startups <- lm(Profit~R.D.Spend+Administration+Marketing.Spend)
summary(model.startups)
install.packages("car")
library(car)

vif(model.startups) # variance inflation factor

model2 <- lm(Profit~Administration+Marketing.Spend)
summary(model2)

n=nrow(startups_new)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test=startups_new[-train,]

pred=predict(model2,newdat=test)
actual=test$Profit
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse

