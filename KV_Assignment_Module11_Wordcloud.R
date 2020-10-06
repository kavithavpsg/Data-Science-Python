install.packages("rvest")
install.packages("XML")
install.packages("magrittr")


library(rvest)
library(XML)
library(magrittr)
library(xml2)

######### Apple watch Reviews ###########

aurl <-"https://www.amazon.in/Apple-Watch-GPS-44mm-Aluminium/dp/B07XWYJDDW/ref=sr_1_13?dchild=1&qid=1593212912&refinements=p_89%3AApple%2Cp_6%3AA14CZOWI0VEHLG%7CAT95IG9ONZD7S&s=electronics&sr=1-13#customerReviews"

reviews <- NULL

for (i in 1:20){
  mac <- read_html(as.character(paste(aurl,i,sep ="=")))
  rev <- mac %>% html_nodes(".review-text") %>% html_text()
  reviews <- c(reviews,rev)
}

write.table(reviews,"applewatch.txt")
getwd()

##################################

txt <- reviews

str(txt)
length(txt)

# Corpus
install.packages("tm")
library(tm)

x <- Corpus(VectorSource(txt))

inspect(x[1])
inspect(x[160])

x <- tm_map(x, function(x) iconv(enc2utf8(x), sub='byte'))
?tm_map

# Data Cleansing
x1 <- tm_map(x, tolower)
inspect(x1[1])

x1 <- tm_map(x1, removePunctuation)
inspect(x1[1])

inspect(x1[5])
x1 <- tm_map(x1, removeNumbers)
inspect(x1[1])

x1 <- tm_map(x1, removeWords, stopwords('english'))
inspect(x1[1])
inspect(x1[3])

# striping white spaces 
x1 <- tm_map(x1, stripWhitespace)
inspect(x1[1])

# Term document matrix 
# converting unstructured data to structured format using TDM

tdm <- TermDocumentMatrix(x1)
tdm
dtm <- t(tdm) # transpose
dtm <- DocumentTermMatrix(x1)

tdm <- as.matrix(tdm)
dim(tdm)

tdm[1:20, 1:20]

inspect(x[3])

# Bar plot
w <- rowSums(tdm)
w

w_sub <- subset(w, w >= 65)
w_sub

barplot(w_sub, las=1, col = rainbow(30))

# Term phone repeats maximum number of times
x1 <- tm_map(x1, removeWords, c('iphone','even','iphone','just','now','read','watch','apple','display','get','series'))
x1 <- tm_map(x1, stripWhitespace)

tdm <- TermDocumentMatrix(x1)
tdm

tdm <- as.matrix(tdm)
tdm[100:109, 1:20]

# Bar plot after removal of the term 'phone'
w <- rowSums(tdm)
w

w_sub <- subset(w, w >= 50)
w_sub
sort(w_sub)

barplot(w_sub, las=2, col = rainbow(30))

##### Word cloud #####
install.packages("wordcloud")
library(wordcloud)

wordcloud(words = names(w_sub), freq = w_sub)

w_sub1 <- sort(rowSums(tdm), decreasing = TRUE)
head(w_sub1)

wordcloud(words = names(w_sub1), freq = w_sub1) # all words are considered

# better visualization
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F,colors=rainbow(30),scale=c(2,0.5),rot.per=0.4)
windows()
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F,colors= rainbow(30),scale=c(3,0.5),rot.per=0.3)
?wordcloud

############# Wordcloud2 ###############
install.packages("wordcloud2")
library(wordcloud2)

w1 <- data.frame(names(w_sub), w_sub)
colnames(w1) <- c('word', 'freq')
?wordcloud2

wordcloud2(w1, size=0.4, shape='circle')

wordcloud2(w1, size=0.3, shape = 'triangle')

############

