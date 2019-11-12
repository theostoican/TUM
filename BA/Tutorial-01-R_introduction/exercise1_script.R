# Business Analytics WS 2019/2020
# Introduction to R - Exercise
# Please write your code below.

setwd("/Users/tstoican/tum-sem1/Ba/Tutorial-01-R_introduction")
getwd()
data <- read_csv("data/LaborSupply1988.csv")
#1.1
glimpse(data)
ncol(data)
nrow(data)
head(data)
#e
range(data$age)
#f
groupedKids <- group_by(data, kids)
summarise(groupedKids, mean_hours=mean(lnhr))
#g
data %>% filter(age == 40) %>% summarise(mean_kids=mean(kids))
mean(filter(data, age == 40)$kids)
#1.2
#a
hist(data$age)
#b
age_kids <- data %>% group_by(age) %>%  summarise(avg_kids=mean(kids))
plot(age_kids$age, age_kids$avg_kids)
cor(age_kids$age, age_kids$avg_kids)
#d
#c
plot(data$age, data$lnwg)
#e
plot(data$lnhr, data$age, pch=as.numeric(data$disab + 1), col=c("blue", "red")[as.numeric(data$disab + 1)])
#f
