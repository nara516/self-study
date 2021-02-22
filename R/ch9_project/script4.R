##연령대 및 성별 월급 차이
gender_income = welfare %>% 
  filter(!is.na(income)) %>% 
  group_by(ageg, gender) %>% 
  summarise(mean_income = mean(income))
gender_income

ggplot(data = gender_income, aes(x = ageg, y = mean_income, fill = gender)) + 
  geom_col() + scale_x_discrete(limits = c("young", "middle", "old"))
#fill : 막대가 성별에 따라 다른 색으로 표현되도록 설정
#scale_x_discrete 를 이용하여 막대를 연령대순으로 설정

#막대분리를 위해 geom_col(position = "dodge") 설정 추가
ggplot(data = gender_income, aes(x = ageg, y = mean_income, fill = gender)) +
  geom_col(position = "dodge") + 
  scale_x_discrete(limits = c("young", "middle", "old"))


##연령대를 구분하지 않고 나이 및 성별 월급 평균표
gender_age = welfare %>% 
  filter(!is.na(income)) %>% 
  group_by(age, gender) %>% 
  summarise(mean_income = mean(income))

head(gender_age)

ggplot(data = gender_age, aes(x = age, y = mean_income, col = gender)) + geom_line()
