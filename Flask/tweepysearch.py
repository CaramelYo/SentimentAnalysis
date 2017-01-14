
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

import tweepy
import re


def twittersearch(q, number):
    # Variables that contains the user credentials to access Twitter API
    consumer_key = 'wtWw3J6IzZhW5vtlNhJ16xfVS'
    consumer_secret = 'rw6G0ErwEEos6Xvl6ua59csCCuGZwkHJZ1IOZ94PAV5OvVHAeq'
    access_token = '3060634116-BRClW1IjisNuVAP1LMphUHSWFx4DWWOBQjfWThE'
    access_token_secret = 'SPZS5erC2jyhVkuxcvvtO1xyOuUupA8WySME9htg9Xy00'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Search for latest tweets about "#python"

    tweets = []
    tweetstime = []
    for statues in tweepy.Cursor(api.search, q=q, lang="en").items(number):
        tweets.append(statues.text)
        tweetstime.append(statues.created_at)
    
    text = []
    for i in range(0, len(tweets), 1):
        x = tweets[i]
        y = re.sub("\\\\n","",x)
        z = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",y).split())
        text.append(z)
        
    return text, tweetstime
    
