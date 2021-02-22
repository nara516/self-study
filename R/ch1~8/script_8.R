# 이상치, 결측치 제거
library(dplyr)

##결측치 정제하기
df = data.frame(gender = c("M", "F", NA, "M", "F"),
                score = c(5, 4, 3, 4, NA))
df

is.na(df)   #결측치 확인 (T/F)
table(is.na(df))
table(is.na(df$gender))
table(is.na(df$score))

df_nomiss = df %>% filter(!is.na(score) & !is.na(gender))      #결측치 제거
df_nomiss

#결측치가 하나라도 있으면 행 전체 제거
df_nomiss2 = na.omit(df)           #omit은 주의해서 사용해야함
df_nomiss2

#함수의 결측치 제외 기능 이용하기
mean(df$score, na.rm = T)    #na.rm 결측치를 제외하고 연산하도록
exam = read.csv("csv_exam.csv")
exam[c(3,8,15), "math"] <- NA         #3,8,15행에 결측치 할당
exam %>% summarise(mean_math = mean(math, na.rm = T))

#결측치 대체하기
exam$math = ifelse(is.na(exam$math), mean(exam$math, na.rm = T), exam$math)
table(is.na(exam$math))

#------------work
mpg = as.data.frame(ggplot2::mpg)
mpg[c(65, 124, 131, 153, 212), "hwy"] = NA

table(is.na(mpg$drv))
table(is.na(mpg$hwy))

mpg %>% filter(!is.na(hwy)) %>% group_by(drv) %>% summarise(mean(hwy, na.rm = T))


##이상치 정제하기
outlier = data.frame(gender = c(1,2,1,3,2,1),
                     score = c(5,4,3,4,2,6))
outlier

outlier$gender = ifelse(outlier$gender == 3, NA, outlier$gender)
outlier$score = ifelse(outlier$score > 5, NA, outlier$score)
outlier
outlier %>% filter(!is.na(gender) & !is.na(score)) %>% 
  group_by(gender) %>% summarise(mean_score = mean(score))


#이상치 제거하기 - 극단적인 값
boxplot(mpg$hwy)
boxplot(mpg$hwy)$stats   #상자 그림 통계치 출력
mpg$hwy = ifelse(mpg$hwy < 12 | mpg$hwy > 37, NA, mpg$hwy)
table(is.na(mpg$hwy))

mpg %>% group_by(drv) %>% summarise(mean_hwy = mean(hwy, na.rm = T))


#------------------work
mpg3 = as.data.frame(ggplot2::mpg)
mpg3[c(10,14,58,93), "drv"] = "k"
mpg3[c(29,43,129,203), "cty"] = c(3,4,39,42)

#mpg3$drv = ifelse(mpg3$drv == "k", NA, mpg3$drv)
mpg3$drv = ifelse(mpg3$drv %in% c("4","f","r"), mpg$drv, NA)
table(mpg3$drv)

boxplot(mpg3$cty)$stats
mpg3$cty = ifelse(mpg3$cty < 9 | mpg3$cty > 26, NA, mpg3$cty)
boxplot(mpg3$cty)$stats
mpg3 %>% filter(!is.na(hwy) & !is.na(cty)) %>% group_by(drv) %>% summarise(mean_cty = mean(cty))
