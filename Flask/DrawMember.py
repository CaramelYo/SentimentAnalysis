from flask import Flask
from nltk.corpus import movie_reviews
import SentimentAnalysis as sa

app = Flask(__name__)

@app.route('/')

def index():
    return "<p>Hello World!</p>" + "<p>" + sa.sentiment(movie_reviews.raw('pos/cv008_29435.txt')) + "</p>"
##    return "<p>Hello World!</p>"

if __name__ == '__main__':
    app.run(port=5001, debug=True)
