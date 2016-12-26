from flask import Flask, render_template

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/sentilive')
def sentilive():
    return render_template("sentilive.html")

@app.route('/sentianalysis')
def sentianalysis():
    return render_template("sentianalysis.html")

@app.route('/graph')
def graph():
    from pandas_datareader import data
    import datetime
    from bokeh.plotting import figure, show, output_file
    from bokeh.embed import components
    from bokeh.resources import CDN

    start = datetime.datetime(2015,11,1)
    end = datetime.datetime.now()

    df = data.DataReader(name="GOOG", data_source = "yahoo", start = start, end = end)

    def inc_dec(c, o):
        if c > o:
            value="Increase"
        elif c < o:
            value="Decrease"
        else:
            value="Equal"
        return value

    df["Status"]=[inc_dec(c,o) for c, o in zip(df.Close,df.Open)]
    df["Middle"]=(df.Open+df.Close)/2
    df["Height"]=abs(df.Close-df.Open)

    p=figure(x_axis_type='datetime', width=1000, height=300, responsive=True)
    p.title="Candlestick Chart"
    p.grid.grid_line_alpha=0.3

    hours_12=12*60*60*1000

    p.segment(df.index, df.High, df.index, df.Low, color="Black")

    p.rect(df.index[df.Status=="Increase"],df.Middle[df.Status=="Increase"],
           hours_12, df.Height[df.Status=="Increase"],fill_color="#CCFFFF",line_color="black")

    p.rect(df.index[df.Status=="Decrease"],df.Middle[df.Status=="Decrease"],
           hours_12, df.Height[df.Status=="Decrease"],fill_color="#FF3333",line_color="black")

    script1, div1, = components(p)
    cdn_js=CDN.js_files[0]
    cdn_css=CDN.css_files[0]
    return render_template("graph.html",
    script1=script1,
    div1=div1,
    cdn_js=cdn_js,
    cdn_css=cdn_css)

@app.route('/references')
def references():
    return render_template("references.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__=="__main__":
    app.run(port=8454, host='140.116.177.150', debug=True)
