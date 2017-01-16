#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:47:52 2017

@author: royzhuang
"""
import pandas as pd
from collections import Counter
from bokeh.charts import output_file, show, Bar
import tweepysearch as search

def barchart(q, since, until, number):
    
    all_text, twittertime, first_text = search.twittersearch(q = q, since = since, until = until, number = number)
    
    count_all = Counter()
    for t in first_text:
        count_all.update(t)

    #for key, value in count_all.most_common(20) :
    #    print key, value

    words = pd.DataFrame.from_dict(count_all.most_common(20), orient='columns', dtype = None)
    words.columns = ['Key Words', 'Words Count']

    #print(words)
    
    a = 'Words Distribution of '
    title = a + q

    bar = Bar(words, 'Key Words',values='Words Count', 
              title = title, legend='top_right', bar_width=0.5)

    output_file("bar_wd.html", title="bar_wd.py example")

    show(bar)


