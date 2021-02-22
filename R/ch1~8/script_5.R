#데이터분석기초(ch.5)
exam = read.csv("csv_exam.csv")

#데이터 파악하기 (함수)
head(exam)
head(exam, 10)

tail(exam)
tail(exam,10)

View(exam)  #뷰어창에서 데이터 보기

dim(exam)   #데이터가 몇행, 몇열로 구성되어 있는지 출력

str(exam)   #변수속성파악하기
#관측치 obs = row

summary(exam)   #요약통계량


#-----------
#mpg데이터 파악하기

#ggplot2의 mpg데이터를 df형태로 불러오기
mpg = as.data.frame(ggplot2::mpg)          #더블콜론을 이용하면 특정패키지에 들어있는 함수나 데이터를 지정할 수 있음

head(mpg)
tail(mpg)
dim(mpg)    #234종에 대한 11개의 변수로 구성
str(mpg)
?mpg  #데이터 설명 (help창에 뜸)
summary(mpg)


#-------------------
#dplyr
install.packages("dplyr")
library(dplyr)

df_raw = data.frame(var1 = c(1,2,1),
                    var2 = c(2,3,2))
df_raw
df_new = df_raw

df_new = rename(df_new, v2 = var2)    #var2를 v2로 수정
df_new
#------------------

#work
#mpg데이터 변수명 바꾸기

mpg2 = mpg
mpg2 = rename(mpg2, city = cty)
mpg2 = rename(mpg2, highway = hwy)
head(mpg2)


#---------------------
#파생변수만들기
df = data.frame(var1 = c(4,3,8),
                var2 = c(2,6,1))
df$var_sum = df$var1 + df$var2         #var_sum 추가
df$var_mean = df$var_sum / 2
df

#work : mpg 통합 연비 변수 만들기
#통합연비변수 : (cty+hwy)/2
mpg$total = (mpg$cty + mpg$hwy)/2
head(mpg)
mean(mpg$total)

#work : 조건문을 활용해 파생변수 만들기

#1. 연비가 기준치를 넘으면 합격, 기준치를 정하기 윟애 total의 평균과 중앙값 확인
summary(mpg$total)

#히스토그램 : 전반적인 분포확인
hist(mpg$total)

mpg$test = ifelse(mpg$total >= 20, "pass", "fail")
head(mpg)

table(mpg$test)    #빈도 테이블 출력

#막대그래프로 빈도 표현하기
library(ggplot2)
qplot(mpg$test)

#work : 중첩 조건문 활용하기
mpg$grade = ifelse(mpg$total >= 30, "A",
                  ifelse(mpg$total >= 20, "B", "C"))
head(mpg)

table(mpg$grade)
qplot(mpg$grade)

#----------------
#work : ggplot2 - midwest 데이터 미국 동북중부 437개 지역의 인구통계 정보

#1. midwest를 데이터 프레임 형태로 불러오기
midwest = as.data.frame(ggplot2::midwest)
head(midwest)
dim(midwest)
str(midwest)
summary(midwest)

#2. 변수명 바꾸기
midwest2 = midwest
midwest2 = rename(midwest2, total = poptotal)
midwest2 = rename(midwest2, asian = popasian)

str(midwest2)

#3. total, asian 변수를 이용해 '전체 인구 대비 아시아 인구 백분율' 파생변수 생성 후 분포살피기
midwest2$asian_per_total = midwest2$asian / midwest2$total * 100
midwest2$asian_per_total
hist(midwest2$asian_per_total)

#4. 아시아 인구 백분율 전체 평균을 구하고 범위에 따른 파생변수
midwest2$asian_per_total_mean = ifelse(midwest2$asian_per_total > mean(midwest2$asian_per_total), "large","small")
midwest2$asian_per_total_mean

#large와 small에 해당하는 지역이 얼마나 되는지 빈도표와 빈도 막대 그래프로 확인
table(midwest2$asian_per_total_mean)
qplot(midwest2$asian_per_total_mean)
