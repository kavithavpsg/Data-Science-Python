corolla <- read.csv("D:/Data science/Module 7-MLR/Assignment/50_Startups (2)/ToyotaCorolla (1).csv")
corolla<-corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
attach(corolla)
summary(corolla)
install.packages("Hmisc")
library(Hmisc)
#Scatter diagram

pairs(corolla)
#Correlation coefficient Matrix - Strength & Direction of correlation
cor(corolla)

# The Linear Model of interest
model.corolla <- lm(Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight)
summary(model.corolla)
install.packages("car")
library(car)

vif(model.corolla) # variance inflation factor

model2 <- lm(Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax)
summary(model2)

n=nrow(corolla)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test=corolla[-train,]

pred=predict(model2,newdat=test)
actual=test$Price
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse


