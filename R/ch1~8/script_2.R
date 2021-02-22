install.packages("ggplot2")
library(ggplot2)
x = c("a","a","b","c")

#빈도막대그래프 출력
qplot(x)

#ggplot의 mpg 데이터로 그래프 만들기
#mpg 데이터 : 미국환경보호국에서 공개한 자료로 자동차 234종의 연비관련 정보

#X축: hwy(자동차가 고속도로에서 1갤런에 몇 마일을 가는지 나타내는 변수) , 고속도로 연비별 빈도 막대 그래프그리기

#data에 mpg, x축에 hwy 변수를 지정하여 그래프 생성
qplot(data = mpg, x = hwy)

#x축 cty
qplot(data = mpg, x = cty)

#x:drv, y:hwy
qplot(data = mpg, x = drv, y = hwy)

#x:drv, y:hwy 선그래프
qplot(data = mpg, x = drv, y = hwy, geom = "line")

#x:drv, y:hwy 상자 그림 형태
qplot(data = mpg, x = drv, y = hwy, geom = "boxplot")

#x:drv, y:hwy 상자 그림 형태, drv별 색 표현
qplot(data = mpg, x = drv, y = hwy, geom = "boxplot", colour = drv)

#함수 메뉴얼 출력
?qplot

#-----------------------------------------
#work
#시험점수 변수 만들고 출력하기
score = c(80,60,70,50,90)
#평균구하기
mean(score)
avg = mean(score)
avg
