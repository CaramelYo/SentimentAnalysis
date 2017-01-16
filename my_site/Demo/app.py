import json
from flask import Flask,request,render_template

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/sentianalysis', methods = ['GET', 'POST'])
def sentianalysis():
    import SentimentAnalysis as sa

    if request.method == 'GET':
        keyWords = request.args.get('keyWords', 'Trump')
        number = request.args.get('number', 10, type=int)
        since = request.args.get('since', '2017-01-15')
        until = request.args.get('until', '2017-01-16')

    else:
        keyWords = request.form['keyWords']
        number = request.form['number']
        since = request.form['since']
        keyWords = request.form['until']

    return sentiUpdate(keyWords, number, since, until)


def sentiUpdate(q, number, since, until):
    from bokeh.layouts import gridplot, WidgetBox
    from bokeh.embed import components
    from bokeh.resources import CDN
    from bokeh.io import vform
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.plotting import figure, output_file, show, curdoc
    from bokeh.models.widgets import TextInput
    import numpy as np

    #for twitter parsing
    import tweepysearch as search

    #for sentiment analysis
    import SentimentAnalysis as sa

    def datetime(x):
        return np.array(x, dtype=np.datetime64)

    #to parse old data from twitter
    #texts, times = search.twittersearch(q = q, number = 10, since = '2017-01-10', until = '2017-01-12')

    sa.printOut('update')
    #texts, times, fullTexts = search.twittersearch(q = q, number = number, since = '2017-01-10', until = '2017-01-16')
    texts, times, fullTexts = search.twittersearch(q = q, number = number, since = since, until = until)
    sa.printOut('search completed')

    result = sa.predictAsDict(texts, number, times)

    p1 = figure(x_axis_type="datetime", title="Sentiment Analysis")
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Score'

    p1.line(datetime(result['date']), result['score'], color='#A6CEE3', legend=q)
    p1.legend.location = 'top_left'

    script1, div2, = components(gridplot([[p1]], plot_width=500, plot_height=500))
    cdn_js=CDN.js_files[0]
    cdn_css=CDN.css_files[0]
    return render_template("sentianalysis.html",
    script1=script1,
    div2=div2,
    cdn_js=cdn_js,
    cdn_css=cdn_css)

@app.route('/graph', methods = ['GET', 'POST'])
def graph():

    if request.method == 'GET':
        keyWords = request.args.get('keyWords', 'Trump')
        number = request.args.get('number', 10, type=int)
        since = request.args.get('since', '2017-01-15')
        until = request.args.get('until', '2017-01-16')

    else:
        keyWords = request.form['keyWords']
        number = request.form['number']
        since = request.form['since']
        keyWords = request.form['until']

    return graphUpdate(keyWords, number, since, until)


def graphUpdate(q, number, since, until):
    from bokeh.layouts import gridplot, WidgetBox
    from bokeh.embed import components
    from bokeh.resources import CDN
    from bokeh.io import vform
    from bokeh.plotting import figure, output_file, show
    from bokeh.models.widgets import TextInput
    import numpy as np
    import pandas as pd
    from collections import Counter
    from bokeh.charts import output_file, show, Bar
    import tweepysearch as search
    import pandas as pd
    import top20bar
    from bokeh.models import CustomJS, ColumnDataSource, Div

    def datetime(x):
        return np.array(x, dtype=np.datetime64)

    #for twitter parsing
    import tweepysearch as search
    texts, times, fullTexts = search.twittersearch(q = q, number = number, since = since, until = until)


    #for bar chart
    all_text, twittertime, first_text = search.twittersearch(q = q, since = since, until = until, number = number)

    count_all = Counter()
    for t in first_text:
        count_all.update(t)

    words = pd.DataFrame.from_dict(count_all.most_common(20), orient='columns', dtype = None)
    words.columns = ['Key Words', 'Words Count']

    a = 'Words Distribution of '
    title = a + q
    wdbar = Bar(words, 'Key Words',values='Words Count', title = title, legend='top_right', bar_width=0.5)
    
    script1, div5, = components(wdbar)
    cdn_js=CDN.js_files[0]
    cdn_css=CDN.css_files[0]
    cdn_js2=CDN.js_files[1]
    cdn_css2=CDN.css_files[1]
    return render_template("graph.html",
    script1=script1,
    div5=div5,
    cdn_js=cdn_js,
    cdn_css=cdn_css,
    cdn_js2=cdn_js2,
    cdn_css2=cdn_css2)


@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__=="__main__":
    app.run(port = 5000, host = '140.116.177.150', debug=True)
