#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 09:41:42 2016

@author: royzhuang
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:07:07 2016

@author: royzhuang
"""

# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import OAuth, TwitterStream
import re

#total key number
keyNumber = 3

def parsetw(track, number):    
    # Variables that contains the user credentials to access Twitter API
    #the key we want to use
    #0 => CYo, 1 => Roy, 2 => %%
    usedKeyNumber = 0

    usedKeys = [] 
    with open('TwitterKeys.txt') as f:
        #to skip title
        f.readline()
        for i in range(usedKeyNumber):
            f.readline()
        usedKeys.append(f.readline().rstrip())
        for i in range(3):
            for j in range(keyNumber):
                f.readline()        
            usedKeys.append(f.readline().rstrip())

    oauth = OAuth(usedKeys[2], usedKeys[3], usedKeys[0], usedKeys[1])

    # Initiate the connection to Twitter Streaming API
    twitter_stream = TwitterStream(auth=oauth)

    # Get a sample of the public data following through Twitter
    iterator = twitter_stream.statuses.filter(track = track, language = "en")

    tweet_count = number
    
    tweets = []

    for tweet in iterator:
        tweet_count -= 1
        # Twitter Python Tool wraps the data returned by Twitter
        # as a TwitterDictResponse object.
        # We convert it back to the JSON format to print/score
        # print json.dumps(tweet)
        tweets.append(json.dumps(tweet))
        if tweet_count <= 0:
            break

    twittertext = []
    hashtags = []


    for i in range(0, len(tweets), 1):
        try:
            x = json.loads(tweets[i])
            if 'text' in x:
                
                twittertext.append(x['text'])
        
                for hashtag in x['entities']['hashtags']:
                    hashtags.append(hashtag['text'])
        except:

            continue
    
    
    text = []
    for i in range(0, len(twittertext), 1):
        x = twittertext[i]
        y = re.sub("\\\\n","",x)
        z = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",y).split())
        text.append(z)

    return text, hashtags
