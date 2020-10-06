cutlet <- read.csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Cutlets.csv")
attach(cutlet)
colnames(cutlet)<-c("UnitA","UnitB")
attach(cutlet)
# Normality test
shapiro.test(UnitA)
# p-value = 0.32 >0.05 so p high null fly => It follows normal distribution

shapiro.test(UnitB)
# p-value = 0.5225 >0.05 so p high null fly => It follows normal distribution

# Variance test
var.test(UnitA,UnitB)
# p-value = 0.3136 > 0.05 so p high null fly => Equal variances

# 2 sample t Test
t.test(UnitA, UnitB, alternative = "two.sided", conf.level = 0.95) 

LabTAT <- read.csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/LabTAT.csv")
View(LabTAT)
colnames(LabTAT) <- c("LabA","LabB","LabC","LabD")
attach(LabTAT)

# Normality test
shapiro.test(`LabA`)
shapiro.test(`LabB`)
shapiro.test(`LabC`)
shapiro.test(`LabD`)

# Variance test
var.test(`LabA`,`LabB`)
var.test(`LabB`,`LabC`)
var.test(`LabC`,`LabA`)
var.test(`LabA`,`LabD`)
var.test(`LabC`,`LabD`)
var.test(`LabB`,`LabD`)

Anova_results <- aov(LabA~LabB+LabC+LabD, data = LabTAT)
summary(Anova_results)

# p-value = 0.166 > 0.05 Accept null hypothesis
# p-value = 0.277 > 0.05 Accept null hypothesis
# p-value = 0.215 > 0.05 Accept null hypothesis
# 4 Labs TAT times are  equal


########### Proportional T Test ##########
library(readxl)

# Load the data: Buyer Ratio data
M_F_Prop_Test<-read.csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Buyer Ratio.csv") 
View(M_F_Prop_Test)

attach(M_F_Prop_Test)

#East
prop.test(x=c(50,550),n=c(393,1731), conf.level = 0.95, alternative = "two.sided")
# two.sided -> means checking for equal proportions of Male and Female under purchased
# p-value = 5.869e-14 < 0.05 accept alternate hypothesis i.e.
# Unequal proportions 

prop.test(x=c(50,550),n=c(393,1731), conf.level = 0.95, alternative = "greater")
# Ha -> Proportions of East > Proportions of observations

#West
prop.test(x=c(142,351),n=c(393,1731), conf.level = 0.95, alternative = "two.sided")
# two.sided -> means checking for equal proportions of Male and Female under purchased
# p-value = 2.835e-11 < 0.05 accept alternate hypothesis i.e.
# Unequal proportions 

prop.test(x=c(142,351),n=c(393,1731), conf.level = 0.95, alternative = "greater")
#p-value = 1.418e-11 < 0.05 check for lesser
prop.test(x=c(142,351),n=c(393,1731), conf.level = 0.95, alternative = "less")
# Ha -> Proportions of West < Proportions of observations

#North
prop.test(x=c(131,480),n=c(393,1731), conf.level = 0.95, alternative = "two.sided")
# two.sided -> means checking for equal proportions of Male and Female under purchased
# p-value = p-value = 0.03126 < 0.05 accept alternate hypothesis i.e.
# Unequal proportions 

prop.test(x=c(131,480),n=c(393,1731), conf.level = 0.95, alternative = "greater")
#p-value = 0.01563 <0.05 - check for less
prop.test(x=c(131,480),n=c(393,1731), conf.level = 0.95, alternative = "less")
#p-value = 0.9844 > 0.05 - Accept Ha
#i.e Proportion of North is less than proportion of observations

prop.test(x=c(70,350),n=c(393,1731), conf.level = 0.95, alternative = "two.sided")
# two.sided -> means checking for equal proportions of Male and Female under purchased
# p-value = p-value = 0.3117 > 0.05 accept null hypothesis i.e.
# Unequal proportions 

prop.test(x=c(70,350),n=c(393,1731), conf.level = 0.95, alternative = "greater")
#p-value = 0.8442
##i.e Proportion of South is less than proportion of observations
prop.test(x=c(70,350),n=c(393,1731), conf.level = 0.95, alternative = "less")
#p-value = 0.1558

######### Chi Square Test ##########

# Load the data: customerorderform.csv
cust_order<-read.csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Customer OrderForm.csv") 
View(cust_order)

attach(cust_order)
table(Country,Defective)

chisq.test(table(Country,Defective))

# p-value = 0.6315 > 0.05  => Accept null hypothesis
# => All countries have equal proportions 

# All Proportions are equal 

Fantaloons <- read.csv("D:/Data science/Module 5-HT/Hypothesis_Testing_Assignment/Faltoons.csv")
attach(Fantaloons)
table(Weekdays,Weekend)
chisq.test(table(Weekdays,Weekend))
