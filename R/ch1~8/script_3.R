#데이터 프레임
#열: 속성(컬럼, 변수) 
#행: 한 사람의 정보 (case)
#데이터가 크다 = 행 또는 열이 많음

eng = c (90,80,60,70)
math = c(50,60,100,20)
df_midterm = data.frame(eng,math)
df_midterm
class = c(1,1,2,2)
df_midterm = data.frame(eng,math,class)
df_midterm

#분석하기
mean(df_midterm$eng)
mean(df_midterm$math)

#데이터 프레임 한번에 만들기
df_midterm = data.frame(eng = c(90,80,60,70),
                        math = c(50,60,100,20),
                        class = c(1,1,2,2))
df_midterm

#work
df = data.frame(제품 = c("사과", "딸기", "수박"),
                가격 = c(1800, 1500, 3000),
                판매량 = c(24, 38, 13))
df
