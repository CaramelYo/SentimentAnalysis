#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:47:52 2017

@author: royzhuang
"""
import re
from nltk.corpus import stopwords
import nltk
import string

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
 
punctuation = list(string.punctuation)

stop = stopwords.words('english') + punctuation + ['RT', 's', 'u2026', 'The', 'amp', 'By', 'points', 
'K', 'You', 'I', 'said', 'https', 'u2019s', '8', 'one', 'want','via', 'This', 'As', 'says', 'would', 'next',
'why', 'u2019', 'u2605', 'A', 'htt', 'ud83d', 'like', '1', 'u2013', 'It']

word_type = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
    
def delete_all(text):
    
    all_text = []
    for t in text:
        terms_stop = [term for term in preprocess(t) if term not in stop]
        pos = nltk.pos_tag(terms_stop)
        terms_stop = [word for word, p in pos if p in word_type]
        #str1 = ' '.join(terms_stop)
        # Update the counter
        all_text.append(terms_stop)
    #print(new_text)
    return all_text

def delete_first(text):
    
    first_text = []
    for t in text:
        terms_stop = [term for term in preprocess(t) if term not in stop]
        #str1 = ' '.join(terms_stop)
        # Update the counter
        first_text.append(terms_stop)
    #print(new_text)
    return first_text