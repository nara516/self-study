library(foreign)
library(dplyr)
library(ggplot2)
library(readxl)

raw_welfare = read.spss(file = "Koweps_hpc10_2015_beta1.sav",
                        to.data.frame = T)
welfare = raw_welfare

head(welfare)
tail(welfare)
View(welfare)
dim(welfare)
str(welfare)
summary(welfare)

#변수이름 바꾸기
welfare =  rename(welfare,
                  gender = h10_g3,      #성별
                  birth = h10_g4,       #태어난 연도
                  marriage = h10_g10,   #혼인상태
                  religion = h10_g11,   #종교
                  income = p1002_8aq1,  #월급
                  code_job = h10_eco9,   #직업코드
                  code_region = h10_reg7) #지역코드


##9-2 성별에 따른 월급 차이 
#1. 변수 검토 및 전처리

class(welfare$gender)    #class : 변수의 타입을 파악
table(welfare$gender)

#이상치 확인
welfare$gender = ifelse(welfare$gender == 9, NA, welfare$gender)
#결측치 확인
table(is.na(welfare$gender))

#성별 항목 이름 부여
welfare$gender = ifelse(welfare$gender == 1, "male", "female")
table(welfare$gender)
qplot(welfare$gender)

#월급 변수 검토 및 전처리
class(welfare$income)
summary(welfare$income)
qplot(welfare$income)
qplot(welfare$income)+xlim(0, 1000)

#코드북 범위에 따른 이상치 확인
summary(welfare$income)

welfare$income = ifelse(welfare$income %in% c(0, 9999), NA, welfare$income)  #이상치 결측처리
table(is.na(welfare$income))  #결측치 확인


###성별에 따른 월급 차이 분석하기
gender_income = welfare %>% 
  filter(!is.na(income)) %>% 
  group_by(gender) %>% 
  summarise(mean_income = mean(income))

gender_income

ggplot(data = gender_income, aes(x = gender, y = mean_income)) + geom_col()




       