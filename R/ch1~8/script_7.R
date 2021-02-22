#work 160p
library(dplyr)
midwest = as.data.frame(ggplot2::midwest)

#1
midwest2 =midwest
head(midwest2) 
midwest2 = midwest2 %>% mutate(popnonage = (poptotal - popadults)/poptotal *100)

#2
midwest2 %>% select(county, popnonage) %>% arrange(desc(popnonage)) %>% head(5)

#3
midwest2 = midwest2 %>% mutate(nonage_grade = ifelse(popnonage > 40, "large",
                                          ifelse(popnonage > 30, "middle", "small")))

table(midwest2$nonage_grade)

#4
midwest2 = midwest2 %>% mutate(asian_ratio = (popasian/poptotal)*100) 

midwest2 %>% select(state, county, asian_ratio) %>% arrange(asian_ratio) %>% head(10)
