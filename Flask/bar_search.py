#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:47:52 2017

@author: royzhuang
"""
import re
import pandas as pd
from collections import Counter
import twittersearch as search
from nltk.corpus import stopwords
import string
from bokeh.charts import output_file, show, Bar

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

    
    
def plotchart(q):
    text, time, tweets = search.twittersearch(q = q , count = 100)

    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['RT', 's', 'u2026', 'The', 'amp', 'By', 'points',
    'K', 'You', 'I', 'said', 'https', 'u2019s', '8', 'one', 'want','via', 'This', 'As', 'says', 'would', 'next',
    'why', 'u2019', 'u2605', 'A', 'htt', 'ud83d', 'like', '1', 'u2013']

    count_all = Counter()
    for i in range(0,len(text),1):

        #terms_all = [term for term in preprocess(tweets[i])]
        terms_stop = [term for term in preprocess(text[i]) if term not in stop]
        # Update the counter
        count_all.update(terms_stop)
# Print the first 5 most frequent words
        print(count_all.most_common(20))


    for key, value in count_all.most_common(20) :
        print key, value

    words = pd.DataFrame.from_dict(count_all.most_common(20), orient='columns', dtype = None)
    words.columns = ['Key Words', 'Words Count']

    print(words)
    
    a = 'Words Distribution of '
    title = a + q

    bar = Bar(words, 'Key Words',values='Words Count', 
              title = title, legend='top_right', bar_width=0.5)

    output_file("bar_search.html", title="bar_search.py example")

    show(bar)


#plotchart(q = 'Hillary')

