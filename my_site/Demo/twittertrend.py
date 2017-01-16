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

#total key number
keyNumber = 3

def trend(number,lag):
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
        x, y, z = search.twittersearch(q = hashtable[i], number = number, since = strCurrent, until = strTomorrow)
        result.append(x)
        time.sleep(lag)

    return result, hashtable
