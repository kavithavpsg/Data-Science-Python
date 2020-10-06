car <- read.csv(file.choose())
car
car <- car[ , -1] 
car
attach(car)
install.packages("moments")
library("moments")
skewness(speed)
skewness(dist)
kurtosis(speed)
kurtosis(dist)
cartopspeed <- read.csv(file.choose())
cartopspeed <- cartopspeed[ , -1]
attach(cartopspeed)
skewness(SP)
skewness(WT)
kurtosis(SP)
kurtosis(WT)
