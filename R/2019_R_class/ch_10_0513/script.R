##8장 나머지 과제 
##6월10일 보강 


install.packages("rJava")
install.packages("memoise")
install.packages("KoNLP")
library(KoNLP)
library(dplyr)
useNIADic()

## 데이터준비하기
txt <- readLines("hiphop.txt")
head(txt)

## 특수문자 제거하기
install.packages("stringr")   #문자처리 패키지
library(stringr) 
txt <- str_replace_all(txt, "\\W", " ")  #특수문자제거

## 명사추출하기
nouns <- extractNoun(txt)

## 단어출현 빈도표 테이블 만들기
## -추출한 명사리스트를 문자열 벡터로 변환 후 빈도표생성
wordcount <- table(unlist(nouns))
df_word <- as.data.frame(wordcount, stringsAsFactors = F)

##-변수명 수정하기
df_word <- rename(df_word, word = Var1, freq = Freq)

##-두 글자 이상 단어 추출하기
df_word <- filter(df_word, nchar(word) >= 2)

##-빈도수 상위 20개 단어 추출
top_20 <- df_word %>% arrange(desc(freq)) %>% head(20)
top_20

##워드클라우드 만들기
install.packages("wordcloud")
library(wordcloud)
library(RColorBrewer)
pal <- brewer.pal(8, "Dark2")
set.seed(1234)

wordcloud(words = df_word$word,
          freq = df_word$freq,
          min.freq = 2,
          max.words = 200,
          random.order = F,
          rot.per = .1,
          scale = c(4,0.3),
          colors = pal)
