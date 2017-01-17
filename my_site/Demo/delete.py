#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:47:52 2017

@author: royzhuang
"""
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
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
sentenceSpliter = ['.', ',', '--']

stop = stopwords.words('english') + punctuation + ['RT', 's', 'u2026', 'The', 'amp', 'By', 'points', 
'K', 'You', 'I', 'said', 'https', 'u2019s', '8', 'one', 'want','via', 'This', 'As', 'says', 'would', 'next',
'why', 'u2019', 'u2605', 'A', 'htt', 'ud83d', 'like', '1', 'u2013', 'It']

word_type = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

backDiscourse = []

def loadBackDiscourse():
    print('loading backDiscourse starts')

    global backDiscourse

    backDiscourse = []
    with open('Vocabulary for Sentiment Analysis/BackDiscourse.txt') as f:
        for line in f:
            backDiscourse.append(word_tokenize(line))

    print('loading backDiscourse ends')

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def removeBackDiscourse(words):
    global backDiscourse
    global sentenceSpliter

    length = len(words)
    isDiscourse = False
    newWords = []
    start = 0
    for i in range(length):
        if(isDiscourse and (words[i] in sentenceSpliter)):
            start = i + 1
            isDiscourse = False
        else:
            for d in backDiscourse:
                if(words[i] == d[0]):
                    dLength = len(d)
                    isDiscourse = True
                    for j in range(1, dLength):
                        if((not words[i + j]) or words[i + j] != d[j]):
                            isDiscourse = False
                            break

                    if(isDiscourse):
                        print('match back discourse :', d, ' ', words[i])
                        newWords.extend(words[start:i])
                        break
            
    if(not isDiscourse):
        newWords.extend(words[start:])

    words = newWords
    del newWords

    return words;
    
def delete_all(text):
    
    all_text = []
    for t in text:
        terms_stop = [term for term in removeBackDiscourse(preprocess(t)) if term not in stop]
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

#init
loadBackDiscourse()