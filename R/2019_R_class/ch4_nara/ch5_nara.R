df_raw <- data.frame(var1 = c(1,2,1),
                     var2 = c(2,3,2))
df_raw

install.packages("dplyr")
library(dplyr)

df_new <- df_raw
df_new <- rename(df_new, v2 = var2)
df_new

df_exam <-data.frame(var1 = c(4,3,8),
                     var2 = c(2,6,1))
df_exam
df_exam$var_sum <-df_exam$var1 + df_exam$var2
df_exam
df_exam$var_mean <- df_exam$var_sum/2
df_exam
summary(df_exam$var_sum)
install.packages("ggplot2")
library(ggplot2)
hist(df_exam$var_sum)

ifelse(df_exam$var_sum >= 8, "pass", "fail")
df_exam$test <- ifelse(df_exam$var_sum >= 8, "pass", "fail")

head(df_exam)
tail(df_exam)
head(df_exam,2)
tail(df_exam,2)
View(df_exam)
dim(df_exam)
str(df_exam)
summary(df_exam)

table(df_exam$test)

qplot(df_exam$test)

df_exam$grade <- ifelse(df_exam$var_sum >= 8, "A", 
                        ifelse(df_exam$var_sum >= 5, "B","C"))

df_exam$grade

head(df_exam)
table(df_exam$grade)
qplot(df_exam$grade)



df_work <- data.frame(midterm = c(30,33,35,29,29,30,30,34,33,30),
                      finals = c(34,22,30,19,24,34,35,35,33,32),
                      report = c(10,10,10,9,9,10,8,10,10,10),
                      attend = c(20,20,18,19,20,20,20,20,19.20))
                      
df_work$sum <- df_work$midterm + df_work$finals + df_work$report + df_work$attend

df_work$grade <- ifelse(df_work$sum >= 90, "A" ,
                        ifelse(df_work$sum >= 80, "B",
                               ifelse(df_work$sum >= 70, "C", "D")))

table(df_work$grade)
qplot(df_work$grade)

write.csv(df_work, file = "df_work.csv")
