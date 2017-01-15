def sen():
    import numpy as np
    from bokeh.io import vform
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.models.widgets import TextInput
    from bokeh.layouts import gridplot
    from bokeh.plotting import figure, show, output_file
    from bokeh.embed import components
    from bokeh.resources import CDN

    import twittertrend

    result, hashtable = twittertrend.trend(number = 10, lag = 0)
    
    print(result)

sen()
