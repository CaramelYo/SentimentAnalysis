# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

import subprocess
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream


def testparse(tra):
    # Variables that contains the user credentials to access Twitter API
    ACCESS_TOKEN = '3060634116-BRClW1IjisNuVAP1LMphUHSWFx4DWWOBQjfWThE'
    ACCESS_SECRET = 'SPZS5erC2jyhVkuxcvvtO1xyOuUupA8WySME9htg9Xy00'
    CONSUMER_KEY = 'wtWw3J6IzZhW5vtlNhJ16xfVS'
    CONSUMER_SECRET = 'rw6G0ErwEEos6Xvl6ua59csCCuGZwkHJZ1IOZ94PAV5OvVHAeq'

    oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    # Initiate the connection to Twitter Streaming API
    twitter_stream = TwitterStream(auth=oauth)

    # Get a sample of the public data following through Twitter
    iterator = twitter_stream.statuses.filter(track = tra, language = "en")

    tweet_count = 5
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
    for i in range(0, tweet_count, 1):
        try:
            tweet1 = json.loads(tweets[i])
            if 'text' in tweet1:
                print tweet1['text']
                twittertext.append(tweet1['text'])
            
                for hashtag in tweet1['entities']['hashtags']:
                    hashtags.append(hashtag['text'])
        except:

            continue

def gcd(m, n):
    if n == 0:
        return m
    else:
        return gcd(n, m % n)
