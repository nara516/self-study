#그래프 만들기
library(ggplot2)
mpg = as.data.frame(ggplot2::mpg)

##산점도 그래프
#x축:displ, y축:hwy로 지정해 배경생성
ggplot(data = mpg, aes(x = displ, y = hwy))
#배경에 산점도 추가
ggplot(data = mpg, aes(x = displ, y = hwy)) + geom_point()
#x축 범위 3~6으로 지정, y축 범위 10~30으로 지정
ggplot(data = mpg, aes(x = displ, y = hwy)) +
  geom_point() + xlim(3,6) + ylim(10,30)

#------------work
ggplot(data = mpg, aes(x = cty, y = hwy)) + geom_point()
midwest = as.data.frame(ggplot2::midwest)
ggplot(data = midwest, aes(x = poptotal, y = popasian)) + geom_point() +
  xlim(0,500000) + ylim(0, 10000)

##막대그래프
#평균막대그래프만들기

#집단별 평균 표 만들기
library(dplyr)
df_mpg = mpg %>% group_by(drv) %>% summarise(mean_hwy = mean(hwy))
#그래프생성하기
ggplot(data = df_mpg, aes(x = drv, y = mean_hwy)) + geom_col()
#크기 순으로 정렬하기
ggplot(data = df_mpg, aes(x = reorder(drv, -mean_hwy), y = mean_hwy)) + geom_col()

#빈도 막대 그래프 만들기  (y축없이 x축만 지정하고 geom_bar()사용)
ggplot(data = mpg, aes(x = drv)) + geom_bar()

#-------work
mpg_cty = mpg %>% filter(class == "suv") %>% group_by(manufacturer) %>% 
  summarise(mean_cty = mean(cty)) %>% arrange(desc(mean_cty)) %>% head(5)
ggplot(data = mpg_cty, aes(x = reorder(manufacturer, -mean_cty), y = mean_cty)) + geom_col()

ggplot(data = mpg, aes(x = class)) + geom_bar()


##선그래프 - 시간에 따라 달라지는 데이터 표현하기 (시계열 그래프)
#시계열 그래프 만들기
economics = as.data.frame(ggplot2::economics)
ggplot(data = economics, aes(x = date, y = unemploy)) + geom_line()

ggplot(data = economics, aes(x = date, y = psavert)) + geom_line()

##상자그림 - 집단 간 분포 차이 표현하기
ggplot(data = mpg, aes(x = drv, y = hwy)) + geom_boxplot()

#-----work
class_mpg = mpg %>% filter(class %in% c("compact", "subcompact", "suv"))

ggplot(data = class_mpg, aes(x = class, y =cty)) + geom_boxplot()
