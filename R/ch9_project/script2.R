#나이와 월급의 관계

#나이변수 검토 및 전처리
class(welfare$birth)
summary(welfare$birth)
qplot(welfare$birth)

table(is.na(welfare$birth))
welfare$birth = ifelse(welfare$birth == 9999, Na, welfare$birth)
table(is.na(welfare$birth))

#파생변수 만들기 - 나이
welfare$age = 2015 - welfare$birth + 1
summary(welfare$age)

qplot(welfare$age)


#나이에 따른 월급 평균표 만들기
#welfare에서 월급변수 결측치 없는것만 필터
#나이에따라 그룹으로 나눔
#그룹(나이)별 월급 평균 변수 추가
age_income = welfare %>% 
  filter(!is.na(income)) %>% 
  group_by(age) %>% 
  summarise(mean_income = mean(income))
head(age_income)

ggplot(data = age_income, aes(x = age, y = mean_income)) + geom_line()




