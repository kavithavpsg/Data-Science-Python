# Load the Dataset
library(readr)
library(readxl)
cococola <- read_xlsx("D:/Data science/Module 20 - Forecast and Time series/CocaCola_Sales_Rawdata (1)/CocaCola_Sales_Rawdata (1).xlsx") # read the cococola data
View(cococola) 

windows()
plot(cococola$Sales, type="o")

# Data Preparation
cococola["t"] <- 1:42
View(cococola)

cococola["t_square"] <- cococola["t"] * cococola["t"]
cococola["log_Sales"] <- log(cococola["Sales"])
library(stringr)
cococola["Quarter"] = str_sub(cococola$Quarter,1,2) 

install.packages('fastDummies')
library('fastDummies')
cococola <- dummy_cols(cococola, select_columns = 'Quarter')

# Partition the time series
train <- cococola[1:30, ]
test <- cococola[31:42, ]


########################### LINEAR MODEL #############################
linear_model <- lm(Sales ~ t, data=train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model,interval='predict', newdata =test))
View(linear_pred)

rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear # 714.0144



######################### Exponential #################################

expo_model <- lm(log_Sales ~ t, data=train)
summary(expo_model)

expo_pred <- data.frame(predict(expo_model, interval='predict', newdata=test))
View(expo_pred)

rmse_expo<-sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo # 552.2821

######################### Quadratic ####################################

Quad_model <- lm(Sales ~ t + t_square, data=train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval='predict', newdata=test))

rmse_Quad <- sqrt(mean((test$Sales - Quad_pred$fit)^2, na.rm=T))
rmse_Quad # 646.2715

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Quarter_Q1+Quarter_Q2+Quarter_Q3+Quarter_Q4, data=train)
summary(sea_add_model)

sea_add_pred <- data.frame(predict(sea_add_model, newdata=test, interval='predict'))

rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add # 1778.007

######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Sales ~ Quarter_Q1+Quarter_Q2+Quarter_Q3+Quarter_Q4, data = train)
summary(multi_sea_model)

multi_sea_pred <- data.frame(predict(multi_sea_model, newdata=test, interval='predict'))

rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea # 1828.924

######################## Additive Seasonality with Linear trend #################

Add_sea_Linear_model <- lm(Sales ~ t+Quarter_Q1+Quarter_Q2+Quarter_Q3+Quarter_Q4,data=train)
summary(Add_sea_Linear_model)
Add_sea_Linear_pred<-data.frame(predict(Add_sea_Linear_model,interval='predict',newdata=test))
rmse_Add_sea_Linear<-sqrt(mean((test$Sales-Add_sea_Linear_pred$fit)^2,na.rm=T))
rmse_Add_sea_Linear # 637.9405

######################## Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model<-lm(Sales ~ t+t_square+Quarter_Q1+Quarter_Q2+Quarter_Q3+Quarter_Q4,data=train)
summary(Add_sea_Quad_model)

Add_sea_Quad_pred<-data.frame(predict(Add_sea_Quad_model,interval='predict',newdata=test))
rmse_Add_sea_Quad<-sqrt(mean((test$Sales-Add_sea_Quad_pred$fit)^2,na.rm=T))
rmse_Add_sea_Quad # 586.0533

######################## Multiplicative Seasonality Linear trend ##########################

multi_add_sea_model<-lm(log_Sales ~ t+Quarter_Q1+Quarter_Q2+Quarter_Q3+Quarter_Q4,data = train)
summary(multi_add_sea_model) 

multi_add_sea_pred<-data.frame(predict(multi_add_sea_model,newdata=test,interval='predict'))
rmse_multi_add_sea<-sqrt(mean((test$Sales-exp(multi_add_sea_pred$fit))^2,na.rm = T))
rmse_multi_add_sea # 410.2497
###### Preparing table on model and it's RMSE values 

table_rmse <- data.frame(c("rmse_linear","rmse_expo","rmse_Quad","rmse_sea_add","rmse_multi_sea","rmse_Add_sea_Quad","rmse_multi_add_sea"),c(rmse_linear,rmse_expo,rmse_Quad,rmse_sea_add,rmse_multi_sea,rmse_Add_sea_Quad,rmse_multi_add_sea))
View(table_rmse)

colnames(table_rmse) <-c ("model","RMSE")
View(table_rmse)

############### Multiplicative Seasonality Linear trend has least RMSE value############################# Additive seasonality with Quadratic Trend has least RMSE value

new_model <- lm(log_Sales ~ t+Quarter_Q1+Quarter_Q2+Quarter_Q3+Quarter_Q4,data = train)
summary(new_model)

resid <- residuals(new_model)
resid[1:30]

# Predict
library(readr)
new_data <- read_csv(file.choose())
View(new_data)

pred_new <- data.frame(predict(new_model, newdata=new_data, interval = 'predict'))
View(pred_new)
pred_new$fit
 