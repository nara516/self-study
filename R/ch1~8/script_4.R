#외부데이터 이용하기
install.packages("readxl")
library(readxl)

df_exam = read_excel("excel_exam.xlsx")
df_exam
mean(df_exam$english)
mean(df_exam$science)

#컬럼명이 없을 때
df_exam_novar = read_excel("excel_exam_novar.xlsx", col_names = F)
df_exam_novar

#엑셀 파일에 시트가 여러개일때
df_exam_sheet =  read_excel("excel_exam_sheet.xlsx", sheet = 3)
df_exam_sheet

#CSV 파일 불러오기
df_csv_exam = read.csv("csv_exam.csv")
df_csv_exam

#데이터프레임을 CSV파일로 저장하기
df_midterm = data.frame(english = c(90,80,60,70),
                        math = c(50,60,100,20),
                        class = c(1,1,2,2))
write.csv(df_midterm, file = "df_midterm.csv")
