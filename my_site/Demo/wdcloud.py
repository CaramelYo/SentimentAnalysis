#!/usr/bin/env python2
"""
Masked wordcloud
================
Using a mask you can generate wordclouds in arbitrary shapes.
"""

#from os import path
#from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
import tweepysearch as search
#d = path.dirname(__file__)

# Read the whole text.
# text = open(path.join(d, 'twitter_data.txt')).read()
def word_cloud(q, number, since, until):
    
    all_text, time , first_text = search.twittersearch(q = q , number = number, since = since, until = until)
    
    
    for t in first_text:
    
        str1 = ' '.join(t)
        print(str1)

    # read the mask image
    # taken from
    # http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
    #alice_mask = imread(path.join(d, "alice_mask.jpg"))

    wc = WordCloud(background_color="white", max_words=2000, mask = None,
               stopwords=STOPWORDS.add("said"))

    # generate word cloud
    wc.generate(str1)

    # store to file
    #wc.to_file(path.join(d, "alice.jpg"))
    plt.imshow(wc)
    plt.axis("off")
    plt.figure()
    #plt.imshow(alice_mask, cmap=plt.cm.gray)
    #plt.axis("off")
    #plt.show()
    plt.savefig('worldcloud.jpeg')