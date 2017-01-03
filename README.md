Parse data from twitter!!


Example : 

(1) If you want to parse specific data by using key word(ex:Trump, 2017, HappyNewYear, etc...), try below:

import twitterstream as stream

result = stream.parsetw(track = 'Trump', number = 100) 

# There are two argument needed. 'track' is the word you are interested and 'number' is how much tweets you want to parse from twitter.


(2) If you want to see top 10 trend in Twitter and observe real-time reviews about those trends, try below:

import twittertrend 

result = twittertrend.trend(number = 5, lag = 5)

# There are two argument needed. 'lag' is the search time in seconds. Because Twitter Trend API has its time limit, you need to appropriately set the time lag to prevent system "break down" !! 'number' is the same as twittterstream.

# The return from twittertrend is combined with reviews of each top 10 trend and table of trend name, such as:

hashtag1 = result[0]  # The reviews of first trend 
hashtag2 = result[1]  # The reviews of second trend
hasgtage = result[10] # The overall table of all trend



