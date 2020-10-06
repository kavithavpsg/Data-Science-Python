# Load the Dataset
library(readr)
library(readxl)
airlines <- read_xlsx("D:/Data science/Module 20 - Forecast and Time series/CocaCola_Sales_Rawdata (1)/Airlines Data Set (1).xlsx") # read the airlines data
View(airlines) 

windows()
plot(airlines$Passengers, type="o")

# Data Preparation
airlines["t"] <- 1:96
View(airlines)

airlines["t_square"] <- airlines["t"] * airlines["t"]
airlines["log_Passengers"] <- log(airlines["Passengers"])
attach(airlines)

# So creating 12 dummy variables 
X <- data.frame(outer(rep(month.abb, length = 96), month.abb,"==") + 0 )# Creating dummies for 12 months
View(X)

colnames(X) <- month.abb # Assigning month names 
View(X)

airlines <- cbind(airlines, X)
View(airlines)

colnames(airlines)

# Partition the time series
train <- airlines[1:70, ]
test <- airlines[71:96, ]


########################### LINEAR MODEL #############################
linear_model <- lm(Passengers ~ t, data=train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model,interval='predict', newdata =test))
View(linear_pred)

rmse_linear <- sqrt(mean((test$Passengers - linear_pred$fit)^2, na.rm = T))
rmse_linear # 48.30986


######################### Exponential #################################

expo_model <- lm(log_Passengers ~ t, data=train)
summary(expo_model)

expo_pred <- data.frame(predict(expo_model, interval='predict', newdata=test))
View(expo_pred)

rmse_expo<-sqrt(mean((test$Passengers - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo # 43.47847

######################### Quadratic ####################################

Quad_model <- lm(Passengers ~ t + t_square, data=train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval='predict', newdata=test))

rmse_Quad <- sqrt(mean((test$Passengers - Quad_pred$fit)^2, na.rm=T))
rmse_Quad # 43.89814

######################### Additive Seasonality #########################

sea_add_model <- lm(Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov, data=train)
summary(sea_add_model)

sea_add_pred <- data.frame(predict(sea_add_model, newdata=test, interval='predict'))

rmse_sea_add <- sqrt(mean((test$Passengers - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add # 124.9757

######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov, data = train)
summary(multi_sea_model)

multi_sea_pred <- data.frame(predict(multi_sea_model, newdata=test, interval='predict'))

rmse_multi_sea <- sqrt(mean((test$Passengers - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea # 129.6291

######################## Additive Seasonality with Linear trend #################

Add_sea_Linear_model <- lm(Passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train)
summary(Add_sea_Linear_model)
Add_sea_Linear_pred<-data.frame(predict(Add_sea_Linear_model,interval='predict',newdata=test))
rmse_Add_sea_Linear<-sqrt(mean((test$Passengers-Add_sea_Linear_pred$fit)^2,na.rm=T))
rmse_Add_sea_Linear # 34.50209

######################## Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model<-lm(Passengers ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train)
summary(Add_sea_Quad_model)

Add_sea_Quad_pred<-data.frame(predict(Add_sea_Quad_model,interval='predict',newdata=test))
rmse_Add_sea_Quad<-sqrt(mean((test$Passengers-Add_sea_Quad_pred$fit)^2,na.rm=T))
rmse_Add_sea_Quad # 30.39304

######################## Multiplicative Seasonality Linear trend ##########################

multi_add_sea_model<-lm(log_Passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data = train)
summary(multi_add_sea_model) 

multi_add_sea_pred<-data.frame(predict(multi_add_sea_model,newdata=test,interval='predict'))
rmse_multi_add_sea<-sqrt(mean((test$Passengers-exp(multi_add_sea_pred$fit))^2,na.rm = T))
rmse_multi_add_sea # 11.72479

###### Preparing table on model and it's RMSE values 

table_rmse <- data.frame(c("rmse_linear","rmse_expo","rmse_Quad","rmse_sea_add","rmse_multi_sea","rmse_Add_sea_Quad","rmse_multi_add_sea"),c(rmse_linear,rmse_expo,rmse_Quad,rmse_sea_add,rmse_multi_sea,rmse_Add_sea_Quad,rmse_multi_add_sea))
View(table_rmse)

colnames(table_rmse) <-c ("model","RMSE")
View(table_rmse)

############### Multiplicative Seasonality Linear trend has least RMSE value############################# Additive seasonality with Quadratic Trend has least RMSE value

new_model <-lm(log_Passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data = train)
summary(new_model)

resid <- residuals(new_model)
resid[1:70]

# Predict
library(readr)
new_data <- read_csv(file.choose())
View(new_data)

pred_new <- data.frame(predict(new_model, newdata=new_data, interval = 'predict'))
View(pred_new)
pred_new$fit
