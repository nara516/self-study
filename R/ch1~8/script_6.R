#ch.6 데이터 가공하기
#dplyr : %>% 기호로 함수들을 나열 (shift + ctrl + M)
library(dplyr)
exam = read.csv("csv_exam.csv")
head(exam)

## filter:행 추출
#class가 1인 경우만 추출
exam %>% filter(class == 1)
#class가 2인 경우만 추출
exam %>% filter(class == 2)
#class가 1이 아닌 경우만 추출
exam %>% filter(class != 1)
#조건을 충족하는 행 추출
exam %>% filter(math > 50)
exam %>% filter(class == 1 & math >= 50)
exam %>% filter(english >= 90 | math >= 50)                
#목록에 해당하는 행 추출하기
exam %>% filter(class == 1 | class == 3 | class == 5)
exam %>% filter(class %in% c(1,3,5))   #1,3,5에 해당하면 추출/ %in%: 변수의 값이 지정한 조건 목록에 해당하는지 (매치연산자)

#-------work
mpg = as.data.frame(ggplot2::mpg)
displ4 = mpg %>% filter(displ < 4)
displ4_avg = mean(displ4$hwy)  
displ4_avg
displ5 = mpg %>% filter(displ > 5)
displ5_avg = mean(displ5$hwy)  
displ5_avg
displ4_avg > displ5_avg

audi = mpg %>% filter(manufacturer == "audi")
audi_avg = mean(audi$cty)
audi_avg
toyota = mpg %>% filter(manufacturer == "toyota")
toyota_avg = mean(toyota$cty)
toyota_avg
audi_avg < toyota_avg

mau_list = mpg %>% filter(manufacturer %in% c("chevrolet","ford","honda"))
hwy_mean = mean(mau_list$hwy)
hwy_mean


## select : 필요한 변수만 추출하기 (열 추출)
exam %>% select(english)
exam %>% select(class, math, english)
#변수 제외하기
exam %>% select(-math)

#fliter와 select 조합하기
exam %>% filter(class == 1) %>% select(english)
head(exam)

exam %>% 
  select(id, math) %>% 
  head

#--------work
mpg = as.data.frame(ggplot2::mpg)

mpg2 = mpg %>% select(class,cty)
str(mpg2)

suv = mpg2 %>% filter(class == "suv")
suv_cty_avg = mean(suv$cty)
suv_cty_avg
compact = mpg2 %>% filter(class == "compact")
compact_cty_avg = mean(compact$cty)
compact_cty_avg

suv_cty_avg < compact_cty_avg


## arrange : 순서대로 정렬하기
#오름차순 정렬
exam %>% arrange(math) %>% head
#내림차순 정렬
exam %>% arrange(desc(math)) %>% head
#-------work
audi = mpg %>% filter(manufacturer == "audi")
audi %>% arrange(desc(hwy)) %>% head(5)


##mutate : 파생변수 추가하기
exam %>% mutate(total = math + english + science) %>% head
exam %>% mutate(total = math + english + science, 
                mean = (math + english + science)/3) %>% head

#mutate에 조건문 적용하기
exam %>% mutate(test = ifelse(science >= 60, "pass", "fail")) %>% head
#dplyr 함수 이어서 사용하기
exam %>% mutate(total = math + english + science) %>% arrange(total) %>% head

#-----------work
mpg3 = as.data.frame(ggplot2::mpg)
mpg3 = mpg3 %>% mutate(sum_cthw = cty + hwy)
mpg3 = mpg3 %>% mutate(mean_cthw = sum_cthw/2)
mpg3 %>% arrange(desc(mean_cthw)) %>% head(3)

mpg4 =mpg3
mpg4 %>% mutate(sum_cthw = cty + hwy, mean_cthw = sum_cthw/2) %>%
  arrange(desc(mean_cthw)) %>% head(3)


## group_by, summarise : 집단별로 요약하기
exam %>% summarise(mean_math = mean(math))

exam %>%  group_by(class) %>% summarise(mean_math = mean(math))

#여러 요약 통계량 한 번에 산출하기
exam %>% group_by(class) %>%
  summarise(mean_math = mean(math),
            sum_math = sum(math),
            median_math = median(math),
            n = n())  #n: 데이터가 몇 행으로 되어 있는지 빈도를 구하는 기능

#각 집단별로 다시 집단 나누기
mpg %>% group_by(manufacturer, drv) %>% 
  summarise(mean_cty = mean(cty)) %>%
  head(10)

#회사별로 suv 자동차의 도시 및 고속도로 통합 연비 평균을 구해 내림차순으로 정렬하고 5위까지 출력
mpg %>%  group_by(manufacturer) %>% 
  filter(class == "suv") %>% 
  mutate(tot = (cty+hwy)/2) %>% 
  summarise(mean_tot = mean(tot)) %>% 
  arrange(desc(mean_tot)) %>% 
  head(5)

#--------------work(150p)
library(dplyr)
mpg = as.data.frame(ggplot2::mpg)
mpg %>% group_by(class) %>% summarise(mean_cty = mean(cty)) %>% arrange(desc(mean_cty)) %>% 
  head(3)
mpg %>% filter(class == "compact") %>% group_by(manufacturer) %>% summarise(count = n()) %>% 
  arrange(desc(count))


##데이터 합치기
test1 = data.frame(id = c(1:5),
                    midterm = c(60,80,70,90,85))
test2 = data.frame(id = c(1:5),
                   final = c(70,83,65,85,80))
test1
test2
total = left_join(test1,test2,by = "id")
total

group_a = data.frame(id = c(1:5),
                     test = c(60,80,70,90,85))
group_b = data.frame(id = c(6:10),
                     test = c(70,83,65,95,80))

group_all = bind_rows(group_a, group_b)
group_all

#------------------work
fuel = data.frame(fl = c("c","d","e","p","r"),
                  price_fl = c(2.35, 2.38, 2.11, 2.76, 2.22),
                  stringsAsFactors = F)
fuel

all = left_join(mpg, fuel, by = "fl")
head(all)

all %>% select(model, fl, price_fl) %>% head(5)
