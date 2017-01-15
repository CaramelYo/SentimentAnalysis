
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:07:07 2016

@author: royzhuang
"""

# Import the necessary package to process data in JSON format

import tweepy
import re
from time import gmtime, strftime


def twittersearch(q, number, since, until):
    # Variables that contains the user credentials to access Twitter API
    consumer_key = 'wtWw3J6IzZhW5vtlNhJ16xfVS'
    consumer_secret = 'rw6G0ErwEEos6Xvl6ua59csCCuGZwkHJZ1IOZ94PAV5OvVHAeq'
    access_token = '3060634116-BRClW1IjisNuVAP1LMphUHSWFx4DWWOBQjfWThE'
    access_token_secret = 'SPZS5erC2jyhVkuxcvvtO1xyOuUupA8WySME9htg9Xy00'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Search for some words"

    tweets = []
    tweetstime = []

    a = since.split(' ')

    b = a[0].split('-')

    c = until.split(' ')

    d = c[0].split('-')


    current_time = strftime("%Y-%m-%d", gmtime())

    current_time = current_time.split('-')


    if(int(d[2]) - int(b[2]) > 7):
    
        for i in range(0, 7, 1):
            print(i)
            print('range > 7')
            print('-----------')
            new_since = current_time[0] + '-' + current_time[1] + '-' + str(int(current_time[2])+i -7)
            new_until = current_time[0] + '-' + current_time[1] + '-' + str(int(current_time[2])+i -6)
        
            print(new_since)
            print(new_until)
            for statues in tweepy.Cursor(api.search, q=q, lang="en", since = new_since, until = new_until).items(number):
                tweets.append(statues.text)
                tweetstime.append(statues.created_at)
            
    elif(int(d[2]) - int(b[2]) > 1):
    
        for i in range(0, int(d[2])-int(b[2]), 1):
            print(i)
            print('range < 7')
            print('-----------')
            new_since = b[0] + '-' + b[1] + '-' + str(int(b[2])+i)
            new_until = b[0] + '-' + b[1] + '-' + str(int(b[2])+1+i)
        
            print(new_since)
            print(new_until)
            for statues in tweepy.Cursor(api.search, q=q, lang="en", since = new_since, until = new_until).items(number):
                tweets.append(statues.text)
                tweetstime.append(statues.created_at)

    else:
    
        for statues in tweepy.Cursor(api.search, q=q, lang="en", since = since, until = until).items(number):
            tweets.append(statues.text)
            tweetstime.append(statues.created_at)
    
    text = []
    for i in range(0, len(tweets), 1):
        x = tweets[i]
        y = re.sub("\\\\n","",x)
        z = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",y).split())
        text.append(z)
        
    return text, tweetstime
    
