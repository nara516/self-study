##연령대에 따른 월급 차이

#연령대 변수 검토 및 전처리하기
welfare= welfare %>% 
  mutate(ageg = ifelse(age < 30, "young",
                       ifelse(age <= 59, "middle", "old")))
table(welfare$ageg)
qplot(welfare$ageg)

#연령대에 따른 월급차이 분석하기
#결측치제외 월급변수 행 추출 한 후 연령대에 따라 그룹화 후 평균구해서 변수추가
ageg_income = welfare %>% 
  filter(!is.na(income)) %>%
  group_by(ageg) %>% 
  summarise(mean_income = mean(income))
ageg_income

ggplot(data = ageg_income, aes(x = ageg, y = mean_income)) + geom_col()

##ggplot은 막대를 변수의 알파벳 순으로 정렬하도록 설정되어있음
#막대를 초,중,노년의 나이순으로 정렬되도록 설정
ggplot(data = ageg_income, aes(x = ageg, y = mean_income)) + geom_col() +
  scale_x_discrete(limits = c("young", "middle", "old"))
