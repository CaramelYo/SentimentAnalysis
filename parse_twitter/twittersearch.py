
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
from twitter import OAuth, Twitter
import re


def twittersearch(q, count):
    # Variables that contains the user credentials to access Twitter API
    ACCESS_TOKEN = '3060634116-BRClW1IjisNuVAP1LMphUHSWFx4DWWOBQjfWThE'
    ACCESS_SECRET = 'SPZS5erC2jyhVkuxcvvtO1xyOuUupA8WySME9htg9Xy00'
    CONSUMER_KEY = 'wtWw3J6IzZhW5vtlNhJ16xfVS'
    CONSUMER_SECRET = 'rw6G0ErwEEos6Xvl6ua59csCCuGZwkHJZ1IOZ94PAV5OvVHAeq'

    oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    # Initiate the connection to Twitter REST API
    twitter = Twitter(auth = oauth)

    # Search for latest tweets about "#python"

    ts = twitter.search.tweets(q=q, result_type = 'recent', lang='en', count = count)


    tweets = []

    tweet_count= count

    for status in ts['statuses']:
        tweet_count -= 1
        tweets.append(json.dumps(status['text']))
        if tweet_count <= 0:
            break
    
    searchtext = []
    for i in range(0, len(tweets), 1):
        x = tweets[i]
        y = re.sub("\\\\n","",x)
        z = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",y).split())
        searchtext.append(z)
        
    return searchtext, tweets
    
