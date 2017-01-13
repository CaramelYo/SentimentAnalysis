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
import twittersearch as search
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
    
    #b = 'twitter_Top_10_trend_'
    #c = '.'
    #d = 'txt'
    result = []
    
    start = time.time()
    
    for i in range(0, 10, 1):
        x, y = search.twittersearch(q = hashtable[i], count = number)
        result.append(x)
        print(i)
        time.sleep(lag)
        #a = str(i)
        #f = b + a + c + d
        #with open(f, 'w') as outfile:
            #json.dump(x, outfile)  
        
    end= time.time()
    
    elapsed = end - start

    print("Trend Time taken:", elapsed, "seconds.")
    
    #time.sleep(lag)
    
    return result, hashtable
    






    

    
