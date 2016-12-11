library(TwitteR2Mongo)

getwd()
setwd("/Users/royzhuang/data/db")

# Specify the two collections to use for storing the tweets
setMongoDbCredentials(type = "hometimeline", collection = "hometimeline", db = "twitter", 
                      url = "mongodb://localhost")
setMongoDbCredentials(type = "filterstream", collection = "filterstream", db = "twitter", 
                      url = "mongodb://localhost")
?setMongoDbCredentials
# Check the connection
hometimeline <- connect2MongoDb("hometimeline")
filterstream1 <- connect2MongoDb("filterstream")

setAPICredentials(key = "wtWw3J6IzZhW5vtlNhJ16xfVS", secret = "rw6G0ErwEEos6Xvl6ua59csCCuGZwkHJZ1IOZ94PAV5OvVHAeq")
makeOAuth()

# create a connection to the collection holding the home timeline
hometimeline <- connect2MongoDb("hometimeline")

# retrieve the most recent tweets via updateHometimeline()
updateHometimeline()

# set digits to 13 in order see the exact unixtimestamp in ms
options(digits = 13)

# interval for the last hour
Sys.setlocale("LC_TIME","en_US.UTF-8")
getInterval()

# interval for the last 24 hours
getInterval(60*24) 

# manually specified interval
timeframe <- c("2016-10-01 18:37:43 GMT","2016-12-09 18:37:43 GMT")
getInterval(inputinterval = timeframe)

topHashtagsInInterval(hometimeline, interval = getInterval(60*24), 20)
topHashtagsInInterval(hometimeline, interval = getInterval(inputinterval = timeframe), 200)

# start the streaming process

startStreaming(timeout = 300)

insertSource("/Users/royzhuang/Desktop/TwitteR2Mongo-master/R/filterstream.R", package="TwitteR2Mongo", functions = "startStreaming")
insertSource("/Users/royzhuang/Desktop/TwitteR2Mongo-master/R/filterstream.R", package="TwitteR2Mongo", functions = "streamingLoop")

# start the streaming process with a specified query_url
startStreaming(60,"https://stream.twitter.com/1.1/statuses/filter.json?track=Trump")

# get a random sample of all tweets
startStreaming(60,"https://stream.twitter.com/1.1/statuses/sample.json")

# stop the streaming process
stopStreaming()

createIntervalVector(c("2016-12-10 05:25:01", "2016-12-10 05:30:01"),5)


timeseries <- createTwitterTimeSeries(filterstream1, c("2016-12-10 05:25:01", "2016-12-10 05:30:01"), 3)

interval <- createIntervalVector(c("2016-12-04 23:00:01", "2016-12-05 08:00:01"),10)
print(plotTweets(timeseries[2:nrow(timeseries),], 20, interval))

rankmoving <- rankTweets(timeseries)
head(rankmoving, n = 10)

counts <- absoluteHashtagCount(timeseries)

interval <- getInterval( inputinterval =  c("2016-11-25 03:00:01", "2016-12-05 04:00:01"))
query <- buildRegexQuery(c("Trump"), timestamp = interval)

data <- extractCoordinates("filterstream", "Trump", 1600)

data <- data[data$user$location != "USA", ]
data <- data[data$user$location != "United States", ]
data <- data[data$user$location != "United States of America", ]
result <- geocodeDataFrame(data)
drawMap(data = result, location = "USA", zoom = 4, fill = TRUE)


filterstream[[1]]


