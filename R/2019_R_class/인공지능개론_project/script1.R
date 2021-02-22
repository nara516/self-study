#데이터 불러오기
library(readxl)
data = read_excel('movies.xlsx', col_names = T) 
data2 = as.data.frame(data)

head(data2)

#개봉한 달에 따른 관객 수 평균을 구하기 위한 데이터 가공
library(dplyr)
movie = data2 %>% select(영화명, 개봉일, 관객수)
head(movie)

movie = movie[complete.cases(movie),]		#결측치 제거

movie01 = movie %>% mutate(month=substr(개봉일,6,7))  #개봉한 달 추출하여 변수추가 
head(movie01)

movie_mean <- data.frame('month','adi_mean', stringsAsFactors = FALSE)
#개봉한 달과 관객수 평균을 							  저장할 데이터프레임 만들기
# 개봉한 월에 따른 평균 관객 수 구하여 데이터프레임에 추가하기
adi01 = subset(movie01, month == '01')
mean01 = c('1월', mean(adi01$관객수))
movie_mean = rbind(movie_mean, mean01)

adi02 = subset(movie01, month == '02')
mean02 = c('2월', mean(adi02$관객수))
movie_mean = rbind(movie_mean, mean02)

adi03 = subset(movie01, month == '03')
mean03 = c('3월', mean(adi03$관객수))
movie_mean = rbind(movie_mean, mean03)

adi04 = subset(movie01, month == '04')
mean04 = c('4월', mean(adi04$관객수))
movie_mean = rbind(movie_mean, mean04)

adi05 = subset(movie01, month == '05')
mean05 = c('5월', mean(adi05$관객수))
movie_mean = rbind(movie_mean, mean05)

adi06 = subset(movie01, month == '06')
mean06 = c('6월', mean(adi06$관객수))
movie_mean = rbind(movie_mean, mean06)

adi07 = subset(movie01, month == '07')
mean07 = c('7월', mean(adi07$관객수))
movie_mean = rbind(movie_mean, mean07)

adi08 = subset(movie01, month == '08')
mean08 = c('8월', mean(adi08$관객수))
movie_mean = rbind(movie_mean, mean08)

adi09 = subset(movie01, month == '09')
mean09 = c('9월', mean(adi09$관객수))
movie_mean = rbind(movie_mean, mean09)

adi10 = subset(movie01, month == '10')
mean10 = c('10월', mean(adi10$관객수))
movie_mean = rbind(movie_mean, mean10)

adi11 = subset(movie01, month == '11')
mean11 = c('11월', mean(adi11$관객수))
movie_mean = rbind(movie_mean, mean11)

adi12 = subset(movie01, month == '12')
mean12 = c('12월', mean(adi12$관객수))
movie_mean = rbind(movie_mean, mean12)

data.frame(movie_mean)
View(movie_mean)

movie_mean <- movie_mean[-1,];movie_mean
colnames(movie_mean) = c("month", "adi_mean")
View(movie_mean)

#시각화
library(ggplot2)
ggplot(data = movie_mean, aes(x=month, y = adi_mean)) + geom_col()