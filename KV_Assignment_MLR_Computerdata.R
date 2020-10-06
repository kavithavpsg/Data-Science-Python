computerdata <- read.csv("D:/Data science/Module 7-MLR/Assignment/50_Startups (2)/Computer_Data (1).csv")
attach(computerdata)
summary(computerdata)
install.packages("Hmisc")
library(Hmisc)
#Scatter diagram

pairs(computerdata)
#Correlation coefficient Matrix - Strength & Direction of correlation
computerdata_new = subset(computerdata,select = -c(cd,multi,premium))
cor(computerdata_new)

# The Linear Model of interest
model.computerdata <- lm(price~speed+hd+ram+screen+ads+trend)
summary(model.computerdata)
install.packages("car")
library(car)

vif(model.computerdata) # variance inflation factor

model2 <- lm(price~speed+ram+screen+ads+trend)
summary(model2)

n=nrow(computerdata_new)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test=computerdata_new[-train,]

pred=predict(model2,newdat=test)
actual=test$price
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse

