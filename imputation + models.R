# Prasenjeet Rathore 110092  
# Diego Rodriguez Martinez 110074 
#Jose Caloca Martinez 110558 

#####################
# Model before imputation
#####################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(stargazer)) install.packages("stargazer", repos = "http://cran.us.r-project.org")
# import dataset
df = read.csv("C:/Users/diego/Documents/ABA PROJECT/beeps_vi_es_csv.csv") 
# select country (Poland) and select variables
df = df %>% filter(country == "Poland") %>% select(c(d2,l1,l2,bmk3a,a6c,n2p,j2,b6)) 
# encode -9 by NA
df = replace(df, df == -9, NA)
# set variable "n2p" as numeric
df$n2p = as.numeric(df$n2p) 
# run model
model = lm(d2 ~ ., data = df)
summary(model)

#####################
# Imputation
#####################

library(mice)
#Loading the dataset
data <- read.csv("/Users/college/Documents/ABA/dataset.csv")
#filtering dataset for selected variable
impute=df[c("d2","l1","l2","bmk3a","a6c","n2p","j2","b6")]
tempData <- mice(impute, method = "cart",maxit = 20, m = 1,seed = 500)

write.csv(tempData, "imputed_dataset.csv")
#####################
# Model after imputation
#####################

# import dataset
df = read.csv("./imputed_dataset.csv") # tienes que poner aqui el path del dataset que tienes

# run model
model = lm(d2 ~ ., data = df)
summary()(model)

stargazer(model, 
          out="sales_model.htm", 
          type="html", 
          dep.var.labels=c("Sales"), 
          covariate.labels = c(
              "Number Permanent Full-Time Employees last FY",
              "Number Permanent Full-Time Employees 3 FY ago",
              "% of Working Capital in Government grants",
              "Screener Size",
              "Total Cost of Sales In Last FY",
              "Senior Management % Time Spent In Dealing With Govt Regulations?",
              "Number Full-Time Employees when the company started operations"
              ))