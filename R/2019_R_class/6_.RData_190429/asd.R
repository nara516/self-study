install.packages("ggplot2")
library(ggplot2)
install.packages("readxl")
library(readxl)
df <- read_excel("data2.xlsx")
df
install.packages("dplyr")
library(dplyr)

df_new <-df %>% group_by(month) %>% summarise(mean_s = mean(total))
df_new$mean_s

ggplot(data = df_new, aes(x = month, y= mean_s)) +geom_col()



ggplot(data = df,aes(x = month, y = total))+ geom_line()

