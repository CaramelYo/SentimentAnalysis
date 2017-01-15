import json
from flask import Flask,request,render_template

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    from bokeh.layouts import WidgetBox
    from bokeh.embed import components
    from bokeh.resources import CDN
    from bokeh.io import vform
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.plotting import figure, output_file, show, curdoc
    from bokeh.models.widgets import TextInput

    x = [x*0.005 for x in range(0, 200)]
    y = x

    source = ColumnDataSource(data=dict(x=x, y=y))

    plot = figure(plot_width=400, plot_height=400)
    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    callback = CustomJS(args=dict(source=source), code="""
            var data = source.get('data');
            var f = cb_obj.get('value')
            console.log(f)
            x = data['x']
            y = data['y']
            for (i = 0; i < x.length; i++) {
                y[i] = Math.pow(x[i], f)
            }
            source.trigger('change');
        """)

    #slider = Slider(start=0.1, end=4, value=1, step=.1, title="power", callback=callback)
    #layout = vform(slider, plot)

    text_input = TextInput(value="1", title="power", callback=callback)
    layout = vform(text_input, plot)


    script1 ,div3, = components(layout)
    cdn_js=CDN.js_files[0]
    cdn_css=CDN.css_files[0]
    cdn_js2=CDN.js_files[1]
    cdn_css2=CDN.css_files[1]
    return render_template("about.html",
    script1=script1,
    div3=div3,
    cdn_js=cdn_js,
    cdn_css=cdn_css,
    cdn_js2=cdn_js2,
    cdn_css2=cdn_css2,)


@app.route('/sentianalysis')
def sentianalysis():
    import numpy as np
    from bokeh.io import vform
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.models.widgets import TextInput
    from bokeh.layouts import gridplot
    from bokeh.plotting import figure, show, output_file
    from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT
    from bokeh.embed import components
    from bokeh.resources import CDN

    def datetime(x):
        return np.array(x, dtype=np.datetime64)

    # graph 1
    p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price'

    p1.line(datetime(AAPL['date']), AAPL['adj_close'], color='#A6CEE3', legend='AAPL')
    p1.line(datetime(GOOG['date']), GOOG['adj_close'], color='#B2DF8A', legend='GOOG')
    p1.line(datetime(IBM['date']), IBM['adj_close'], color='#33A02C', legend='IBM')
    p1.line(datetime(MSFT['date']), MSFT['adj_close'], color='#FB9A99', legend='MSFT')
    p1.legend.location = "top_left"



    # graph 2
    #Creating label annotations for spans

    #Importing libraries
    from bokeh.plotting import figure
    from bokeh.io import output_file, show
    from bokeh.sampledata.periodic_table import elements
    from bokeh.models import Range1d, PanTool, ResetTool, HoverTool, ColumnDataSource, LabelSet
    from bokeh.models.annotations import Span, BoxAnnotation, Label, LabelSet

    #Remove rows with NaN values and then map standard states to colors
    elements.dropna(inplace=True) #if inplace is not set to True the changes are not written to the dataframe
    colormap={'gas':'yellow','liquid':'orange','solid':'red'}
    elements['color']=[colormap[x] for x in elements['standard state']]

    #Create three ColumnDataSources for elements of unique standard states
    gas=ColumnDataSource(elements[elements['standard state']=='gas'])
    liquid=ColumnDataSource(elements[elements['standard state']=='liquid'])
    solid=ColumnDataSource(elements[elements['standard state']=='solid'])


    #Create the figure object
    f=figure(title="Stock Closing Prices")

    #adding glyphs
    f.circle(x="atomic radius", y="boiling point",
             size=[i/10 for i in gas.data["van der Waals radius"]],
             fill_alpha=0.2,color="color",legend='Gas',source=gas)

    f.circle(x="atomic radius", y="boiling point",
             size=[i/10 for i in liquid.data["van der Waals radius"]],
             fill_alpha=0.2,color="color",legend='Liquid',source=liquid)

    f.circle(x="atomic radius", y="boiling point",
             size=[i/10 for i in solid.data["van der Waals radius"]],
             fill_alpha=0.2,color="color",legend='Solid',source=solid)

    #Add axis labels
    f.xaxis.axis_label="Atomic radius"
    f.yaxis.axis_label="Boiling point"

    #Calculate the average boiling point for all three groups by dividing the sum by the number of values
    gas_average_boil=sum(gas.data['boiling point'])/len(gas.data['boiling point'])
    liquid_average_boil=sum(liquid.data['boiling point'])/len(liquid.data['boiling point'])
    solid_average_boil=sum(solid.data['boiling point'])/len(solid.data['boiling point'])

    #Create three spans
    span_gas_average_boil=Span(location=gas_average_boil, dimension='width',line_color='yellow',line_width=2)
    span_liquid_average_boil=Span(location=liquid_average_boil, dimension='width',line_color='orange',line_width=2)
    span_solid_average_boil=Span(location=solid_average_boil, dimension='width',line_color='red',line_width=2)

    #Add spans to the figure
    f.add_layout(span_gas_average_boil)
    f.add_layout(span_liquid_average_boil)
    f.add_layout(span_solid_average_boil)

    #Add labels to spans
    label_span_gas_average_boil=Label(x=80, y=gas_average_boil, text="Gas average boiling point", render_mode="css",
                                     text_font_size="10px")
    label_span_liquid_average_boil=Label(x=80, y=liquid_average_boil, text="Liquid average boiling point", render_mode="css",
                                        text_font_size="10px")
    label_span_solid_average_boil=Label(x=80, y=solid_average_boil, text="Solid average boiling point", render_mode="css",
                                       text_font_size="10px")

    #Add labels to figure
    f.add_layout(label_span_gas_average_boil)
    f.add_layout(label_span_liquid_average_boil)
    f.add_layout(label_span_solid_average_boil)

    script1, div2, = components(gridplot([[p1,f]], plot_width=500, plot_height=500))
    cdn_js=CDN.js_files[0]
    cdn_css=CDN.css_files[0]
    return render_template("sentianalysis.html",
    script1=script1,
    div2=div2,
    cdn_js=cdn_js,
    cdn_css=cdn_css)

@app.route('/graph')
def graph():
    import re
    import pandas as pd
    from collections import Counter
    import searchtwitter as search
    from nltk.corpus import stopwords
    import string
    from bokeh.charts import output_file, show, Bar
    from bokeh.embed import components
    from bokeh.resources import CDN
    from bokeh.layouts import WidgetBox
    from bokeh.io import vform
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.plotting import figure, output_file, show, curdoc
    from bokeh.models.widgets import TextInput

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
            #print(count_all.most_common(20))


        for key, value in count_all.most_common(20) :
            print(key, value)

        words = pd.DataFrame.from_dict(count_all.most_common(20), orient='columns', dtype = None, data=dict(x="Key Words", y="Words Count"))
        source = ColumnDataSource(data=dict(x=Key Words, y=Words Count))
        words.columns = ['Key Words', 'Words Count']

        a = 'Words Distribution of '
        title = a + q

        bar = Figure(plot_width=400, plot_height=400)
        bar.Bar(source, 'Key Words',values='Words Count',
                  title = title, legend='top_right', bar_width=0.5)


    callback = CustomJS(args=dict(source=source), code="""
        var data = source.get('data');
        var f = cb_obj.get('value')
        console.log(f)
        x = data['x']
        y = data['y']
        for (i = 0; i < x.length; i++) {
            y[i] = Math.pow(x[i], f)
        }
        source.trigger('change');
    """)

        #print(words)

    #slider = Slider(start=0.1, end=4, value=1, step=.1, title="power", callback=callback)
    #layout = vform(slider, plot)

    text_input = TextInput(value="", title="power", callback=callback)
    layout = vform(text_input, bar)


    script1 ,div5, = components(layout)
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

@app.route('/references')
def references():
    return render_template("references.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__=="__main__":
    app.run(debug=True)
