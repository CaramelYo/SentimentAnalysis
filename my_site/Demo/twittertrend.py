#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 23:23:26 2016

@author: royzhuang
"""

# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth
#import twittersearch as search
import tweepysearch as search
import datetime
import time


def trend(number,lag):
# Variables that contains the user credentials to access Twitter API 
    ACCESS_TOKEN = '3060634116-BRClW1IjisNuVAP1LMphUHSWFx4DWWOBQjfWThE'
    ACCESS_SECRET = 'SPZS5erC2jyhVkuxcvvtO1xyOuUupA8WySME9htg9Xy00'
    CONSUMER_KEY = 'wtWw3J6IzZhW5vtlNhJ16xfVS'
    CONSUMER_SECRET = 'rw6G0ErwEEos6Xvl6ua59csCCuGZwkHJZ1IOZ94PAV5OvVHAeq'

    oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    twitter = Twitter(auth=oauth)

#world_trends = twitter.trends.available(_woeid=1)

    usa_trends = twitter.trends.place(_id = 23424977)


    #x = json.dumps(usa_trends, indent = 4)

    data = usa_trends[0] 
# grab the trends
    trends1 = data['trends']
# grab the name from each trend
    names = [trend['name'] for trend in trends1]
# put all the names together with a ' ' separating them
    #trends1Name = ' '.join(names)
    print('---------------------------------')
    print('test parse tweets for top 10 trend')
    print('---------------------------------')

    hashtag = '#'
    hashtable = []

    for i in range(0, len(names), 1):
    
        if hashtag in names[i]:
            y = names[i]
            print('name'+ str(i) + 'has hashtag:' + y.split('#')[1])
            hashtable.append(y.split('#')[1])
    
    result = []
    current = datetime.datetime.now()
    strCurrent = datetime.datetime.strftime(current, '%Y-%m-%d')
    tomorrow = current + datetime.timedelta(days = 1)
    strTomorrow = datetime.datetime.strftime(tomorrow, '%Y-%m-%d')

    for i in range(0, 10, 1):
        x, y = search.twittersearch(q = hashtable[i], number = number, since = strCurrent, until = strTomorrow)
        result.append(x)
        time.sleep(lag)
    
    return result, hashtable
