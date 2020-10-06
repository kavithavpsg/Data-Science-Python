library(readxl)
###Read data from excel
Assignment_module02 <- read_excel("D:/Data science/Module 2 - EDA/Hands On_Module 2/Assignment_module02.xlsx")
attach(Assignment_module02)

#calculate first moment
mean(Points)
mean(Score)
mean(Weigh)
median(Points)
median(Score)
median(Weigh)

vpoints<-c(Points)
vscore<-c(Score)
vweigh<-c(Weigh)

###Function to calculate mode
mode<-function(x){
  a=table(x)
  return(a[which.max(a)])
}
mode(vpoints)
mode(vscore)
mode(vweigh)  

##Calculate second moment
  
var(Points)
var(Score)
var(Weigh)

sd(Points)
sd(Score)
sd(Weigh)

rangePoints<-max(Points)-min(Points)
rangeScore<-max(Score)-min(Score)
rangeWeigh<-max(Weigh)-min(Weigh)

##Calculate third moment
install.packages("moments")
library(moments) 
skewness(Points)
skewness(Score)
skewness(Weigh)

kurtosis(Points)
kurtosis(Score)
kurtosis(Weigh)

###Pearson correlation coefficient
cor(Points,Weigh)
cor(Score,Weigh)


###Expected weight of patients
wt_of_patients <- c(308, 330, 323, 334, 335, 345, 367, 387, 399)
Exp_wt_patient<-mean(wt_of_patients)
Exp_wt_patient


##Read company data
company<-read.csv("D:/Data science/Module 2 - EDA/Company.csv")
company
attach(company)

##Draw scatter plot
plot(Name.of.company,Measure.X,xlab="NameofCompany",ylab="GrowthinPercentage",main="Company Growth Percentage",las=2)

##Calculate mean
meancompany<-mean(Measure.X)
meancompany

##Calculate standard deviation
sdcompany<-sd(Measure.X)
sdcompany

##Calaculate variance
varcompany<-var(Measure.X)
varcompany





