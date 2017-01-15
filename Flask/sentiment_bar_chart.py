from SentimentAnalysis import *
from tweepysearch import *
from bar_search import *
from twittersearch import * 
from twitterstream import *
from twittertrend import *
from bokeh.layouts import layout, widgetbox
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.io import vform
from bokeh.models import CustomJS, ColumnDataSource, Div
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models.widgets import TextInput
import pandas as pd
from os.path import dirname, join
import numpy as np
from bokeh.io import curdoc


def get_tweets():
    keywords_tweets_val = keywords_tweets.value.strip()
    return keywords_tweets_val


def update():
    #df = get_twitter()
    print('update')
    df2 = get_tweets()
    get_twitter(df2)
#    source.data = dict(
#        x=df['sentiment'],
#        y=df['score'],
#    )

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)

# search tweets & count sentiment

def get_twitter(q = 'Trump'):
    texts, tweets = search.twittersearch(q = [q], number = 10)
    
    result = predict(texts)
    print('result :')
    print(result)
    
    resultList = []
    for key in result.keys():
        resultList.append((key, result[key]))
    
    print('result list:')
    print(resultList)
    
    # plot bar chart
        
    words = pd.DataFrame.from_dict(resultList, orient='columns', dtype = None)
    print(words)
    words.columns = ['sentiment', 'score']
        
    a = 'Words Distribution of '
    title = a
    
    
    # Create Input controls
    keywords_tweets = TextInput(title="The keywords of tweets")
    
    bar = Bar(words, 'sentiment',values='score',
              title = title, legend='top_right', bar_width=0.5)
    
    
    controls = [keywords_tweets]
    for control in controls:
        control.on_change('value', lambda attr, old, new: update())
    
    sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
    
    layout = vform(keywords_tweets, bar)
    show(layout)
    #output_file("barsa.html", title="barsa example")
    
    inputs = widgetbox(*controls, sizing_mode=sizing_mode)
    
    l = layout


# Create Column Data Source that will be used by the plot
'''
source = ColumnDataSource(data=dict(x=words['sentiment'], y=words['score']))
#source = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})

p = figure(plot_height=600, plot_width=700, title="")
bar = p.Bar(words, 'sentiment',values='score',
          title = title, legend='top_right', bar_width=0.5)

#p.Bar(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")
'''



update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Movies"