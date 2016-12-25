setwd("D:/senti/")

install.packages("twitteR")
install.packages("ROAuth")

library(twitteR)
library(ROAuth)

consumerKey <- "HIe2XrWNqcHX1KsoHF1HWkmcV"
consumerSecret <- "SImYk27PgiE6ockPjMbKjtahfzztZDv2N3mtRFh1GdemUT5sm7"
accessToken <- "841573177-LVleaagDeAk8k1lvnvVFjL9aJ7QsNTIPsXf4V3CU"
accessTokenSecret <- "u9L1nX6Vp5Hg1XRBhDgwwlxzJq96pUdiruf1nzOFXfpE8"

setup_twitter_oauth(
  consumerKey,
  consumerSecret,
  accessToken,
  accessTokenSecret )


tweets <- userTimeline("RDataMining", n =3200)

url <- "http://www.rdatamining.com/data/RDataMining-Tweets-20160212.rds"

download.file(url, destfile = "./RDataMining-Tweets-20160212.rds")

tweets2 <- readRDS("./RDataMining-Tweets-20160212.rds")

(n.tweet <- length(tweets))

(n.tweet <- length(tweets2))

# convert tweets to a data frame
tweets.df <- twListToDF(tweets)
tweets2.df <- twListToDF(tweets2)

tweets.df[190, c("id", "created", "screenName", "replyToSN", "favoriteCount", "retweetCount", 
                  "longitude", "latitude", "text")]

# print tweet2 #190 and make text fit for slide width
writeLines(strwrap(tweets.df$text[190], 60))

###  Text Cleaning  ###
install.packages("tm")
library(tm)

# build a corpus, and specify the source to be character vectors
myCorpus <- Corpus(VectorSource(tweets.df$text))

# convert to lower case
myCorpus <- tm_map(myCorpus, content_transformer(tolower))

# remove URLs
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeURL))

# remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct))

# remove stopwords
myStopwords <- c(setdiff(stopwords('english'), c("r", "big")),
                 "use", "see", "used", "via", "amp")
myCorpus <- tm_map(myCorpus, removeWords, myStopwords)

# remove extra whitespace
myCorpus <- tm_map(myCorpus, stripWhitespace)

# keep a copy for stem completion later
myCorpusCopy <- myCorpus

install.packages("SnowballC")
### Stemming and Stem Completion  ###

myCorpus <- tm_map(myCorpus, stemDocument) # stem words
writeLines(strwrap(myCorpus[[190]]$content, 60))

stemCompletion2 <- function(x, dictionary) {
  x <- unlist(strsplit(as.character(x), " "))
  x <- x[x != ""]
  x <- stemCompletion(x, dictionary=dictionary)
  x <- paste(x, sep="", collapse = " ")
  PlainTextDocument(stripWhitespace(x))
}
myCorpus <- lapply(myCorpus, stemCompletion2, dictionary=myCorpusCopy)
myCorpus <- Corpus(VectorSource(myCorpus))
writeLines(strwrap(myCorpus[[190]]$content, 60))

### Issues in Stem Completion: "Miner" vs "Mining"  ###

# count word frequence
wordFreq <- function(corpus, word) {
  results <- lapply(corpus,
                    function(x) {grep(as.character(x), pattern=paste0("\\<", word)) }
                    )
  sum(unlist(results))
}
n.miner <- wordFreq(myCorpusCopy, "miner")
n.mining <- wordFreq(myCorpusCopy, "mining")
cat(n.miner, n.mining)

# replace oldword with newword
replaceWord <- function(corpus, oldword, newword) {
  tm_map(corpus, content_transformer(gsub),
         pattern=oldword, replacement=newword)
}

myCorpus <- replaceWord(myCorpus, "miner", "mining")
myCorpus <- replaceWord(myCorpus, "universidad", "university")
myCorpus <- replaceWord(myCorpus, "scienc", "science")

### Build Term Document Matrix ###

tdm <- TermDocumentMatrix(myCorpus, control = list(wordLengths = c(1, Inf)))
tdm

idx <- which(dimnames(tdm)$Terms %in% c("r", "data", "mining"))
as.matrix(tdm[idx, 21:30])

### Top Frequent Terms ###

install.packages("ggplot2")
library(ggplot2)

(freq.terms <- findFreqTerms(tdm, lowfreq = 20))

term.freq <- rowSums(as.matrix(tdm))
term.freq <- subset(term.freq, term.freq >= 20)
df <- data.frame(term = names(term.freq), freq = term.freq)

library(ggplot2)
ggplot(df, aes(x=term, y=freq)) + geom_bar(stat = "identity") + 
  xlab("Terms") + ylab("Count") + coord_flip() +
  theme(axis.text=element_text(size=7))

### Wordcloud ###

m <- as.matrix(tdm)
# calculate the fequency of words and sort it by fequency
word.freq <- sort(rowSums(m), decreasing = T)
# colors
library(RColorBrewer)
pal <- brewer.pal(9, "BuGn")[-(1:4)]

# plot word cloud
install.packages("wordcloud")
library(wordcloud)
wordcloud(words = names(word.freq), freq = word.freq, min.freq=3,
          random.order = F, colors = pal)
### Associations ###

# which words are associated with 'r'?
findAssocs(tdm, "r", 0.2)

# which words are associated with 'data'?
findAssocs(tdm, "data", 0.2)

### Network of Terms ###
source("https://bioconductor.org/biocLite.R")
biocLite("graph")
install.packages("graph")
library(graph)
source("http://bioconductor.org/biocLite.R")
biocLite("Rgraphviz")
plot(tdm, term=freq.terms, corThreshold = 0.1, weighting = T)

### Topic Modelling ###
dtm <- as.DocumentTermMatrix(tdm)
install.packages("topicmodels")
library(topicmodels)
lda <- LDA(dtm, k = 8)  # find 8 topics
term <- terms(lda, 7) # first 7 terms of evry topic
(trem <- apply(term, MARGIN =2, paste, collapse = ", "))

install.packages("data.table")
library(data.table)
topics <- topics(lda)  # 1st topic identified for every document(tweets2)
topics <- data.frame(data=as.IDate(tweets.df$created), topic = topics)
ggplot(topics, aes(data, fill = term[topic])) + geom_density(position = "stack")


### Sentiment Analysis ###
install.packages("devtools")
require(devtools)
install_github("sentiment140", "okugami79")

# sentiment analysis # 
library(sentiment)
sentiments <- sentiment(tweets.df$text)
table(sentiments$polarity)

# sentiment plot
sentiments$score <- 0
sentiments$score[sentiments$polarity == "positive"] <- 1
sentiments$score[sentiments$polarity == "negative"] <- -1
sentiments$date <- as.IDate(tweets.df$created)
result <- aggregate(score ~ date, data = sentiments, sum)
plot(result, type = "l")

# Retrieve User Info and Followers

user <- getUser("RDataMining")
user$toDataFrame()
friends <-  user$getFriends()   # who this user follows
followers <- user$getFollowers()  # this user's followers
followers2 <- followers[[1]]$getFollowers()  # a follower's followers

source("http://biostat.jhsph.edu/~jleek/code/twitterMap.R")
twitterMap("simplystats")
data("world.cities")
world.cities[,1]
str(world.cities)

grep("Baltimore", world.cities[,1])

twitterMap("simplystats", userLocation = "Baltimore")
quartz()
??quartz

twitterMap("simplystats", plotType = "both")
worldMapEnv
