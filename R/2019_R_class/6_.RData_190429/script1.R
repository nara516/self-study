install.packages("ggplot2")
library(ggplot2)
install.packages("readxl")
library(readxl)
df <- read_excel("data.xlsx")
df

#1.배경설정하기
ggplot(data = df, aes(x = 나이, y = 희망소득))

#2.그래프 추가하기
ggplot(data = df, aes(x = 나이, y = 희망소득)) + geom_point()

#3.축범위 설정
ggplot(data = df, aes(x = 나이, y = 희망소득)) + geom_point() + xlim(22,25) +  ylim(5000, 8000)

#막대그래프(집단 간 차이 표현, 성별 소득 등)
avg <- mean(df$희망소득)
avg
ggplot(data = df, aes(x = 이름, y = 희망소득)) +geom_col()

install.packages("dplyr")
library(dplyr)
df_new <-df %>% group_by(나이) %>% summarise(mean_s = mean(희망소득))
df_new$mean_s

ggplot(data = df_new, aes(x = 나이, y= mean_s)) +geom_col()

ggplot(data = df_new, aes(x = reorder(나이, -mean_s), y=mean_s))+geom_col()


ggplot(data = df, aes(x = 나이)) + geom_bar()

#선그래프
ggplot(data = df_new, aes(x = 나이, y = 
