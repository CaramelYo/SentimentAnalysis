# Web Link
http://luffy.ee.ncku.edu.tw:8454/

Parse data from twitter!!


Example : 

(1) If you want to parse specific data by using key word(ex:Trump, 2017, HappyNewYear, etc...), try below:

import twitterstream as stream

result = stream.parsetw(track = 'Trump', number = 100) 

# There are two argument needed. 'track' is the word you are interested and 'number' is how much tweets you want to parse from twitter.

# Note here that twitterstream parse new tweets, not old tweets.


(2) If you want to see top 10 trend in Twitter and observe real-time reviews about those trends, try below:

import twittertrend 

reviews, hashtable = twittertrend.trend(number = 5, lag = 5)

# There are two argument needed. 'lag' is the search time in seconds. Because Twitter Trend API has its time limit, you need to appropriately set the time lag to prevent system "break down" !! 'number' is the same as twittterstream.

# The return from twittertrend is combined with reviews of each top 10 trend and table of trend names.

(3) If you wnat to parse old tweets from Twitter, try below:

import twittersearch as search

texts, tweets = search.twittersearch(q = ['War','Russia','United States'] , count = 100)

## There are two argument needed. 'q' is the argument that you are interested. You can use more than one word to find that tweets have those words. 'count' is amount of tweets what you want to parse.

## The return from twittersearch is combined with texts, which has been cleaned, of what you interested in and raw texts from tweets, such as:

raw text -> RT @NFL: Kearse had the TD...\nBut @DougBaldwinJr jumped in front &amp; grabbed it with one hand! \n\n\ud83e\udd14 #DETvsSEA #NFLPlayoffs https://t.co/AWU8vZy\u2026 

cleaned text -> RT Kearse had the TD But jumped in front amp grabbed it with one hand ud83e udd14 DETvsSEA NFLPlayoffs


# Part for SentimentAnalysis
