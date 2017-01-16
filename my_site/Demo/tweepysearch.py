
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:07:07 2016

@author: royzhuang
"""

# Import the necessary package to process data in JSON format

import tweepy
import re
import datetime
import delete as de

def twittersearch(q, number, since, until):
    # Variables that contains the user credentials to access Twitter API
    consumer_key = 'wtWw3J6IzZhW5vtlNhJ16xfVS'
    consumer_secret = 'rw6G0ErwEEos6Xvl6ua59csCCuGZwkHJZ1IOZ94PAV5OvVHAeq'
    access_token = '3060634116-BRClW1IjisNuVAP1LMphUHSWFx4DWWOBQjfWThE'
    access_token_secret = 'SPZS5erC2jyhVkuxcvvtO1xyOuUupA8WySME9htg9Xy00'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    tweets = []
    tweetstime = []
    
    since = datetime.datetime.strptime(since, '%Y-%m-%d')
    until = datetime.datetime.strptime(until, '%Y-%m-%d')
    
    current = datetime.datetime.now()
    lastweek = current - datetime.timedelta(days = 7)
    #lt = datetime.datetime.strftime(lastweek,'%Y-%m-%d')
    
    differ = until - since
    
    if(differ.days > 7):
    
        for i in range(0, 7, 1):
            #print(i)
            #print('range > 7')
            #print('-----------')
            new_since = current[0] + '-' + current[1] + '-' + str(int(current[2])+i -7)
            new_until = current[0] + '-' + current[1] + '-' + str(int(current[2])+i -6)
            a = i + 1
            new_since = lastweek + datetime.timedelta(days = a)
            new_since = datetime.datetime.strftime(new_since, '%Y-%m-%d')
            b = i + 2
            new_until = lastweek + datetime.timedelta(days = b)
            new_until = datetime.datetime.strftime(new_until, '%Y-%m-%d')
            #print(new_since)
            #print(new_until)
            #print(new_since)
            #print(new_until)
            for statues in tweepy.Cursor(api.search, q=q, lang="en", since = new_since, until = new_until).items(number):
                tweets.append(statues.text)
                tweetstime.append(statues.created_at)
    elif(differ.days > 1):
    
        for i in range(0, differ.days, 1):
            #print(i)
            #print('range <= 7')
            #print('-----------')

            new_since = since + datetime.timedelta(days = i)
            new_since = datetime.datetime.strftime(new_since, '%Y-%m-%d')
            a = i + 1
            new_until = since + datetime.timedelta(days = a)
            new_until = datetime.datetime.strftime(new_until, '%Y-%m-%d')
            
            #print(new_since)
            #print(new_until)
            for statues in tweepy.Cursor(api.search, q=q, lang="en", since = new_since, until = new_until).items(number):
                tweets.append(statues.text)
                tweetstime.append(statues.created_at)

    elif(0 < differ.days <= 1):
        #print('range <= 1 day')
        for statues in tweepy.Cursor(api.search, q=q, lang="en", since = since, until = until).items(number):
            tweets.append(statues.text)
            tweetstime.append(statues.created_at)
    else:
        print('error occur!! Please check your date format. It must be YYYY/mm/dd.')
    
    text = []
    for i in range(0, len(tweets), 1):
        x = tweets[i]
        y = re.sub("\\\\n","",x)
        z = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",y).split())
        text.append(z)
        
    all_text = de.delete_all(text)
    first_text = de.delete_first(text)
        
    return all_text, tweetstime, first_text
    
