from flask import Flask
from nltk.corpus import movie_reviews
import twittersearch as search
import twittertrend
import twitterstream as stream
import SentimentClassifier as sc

app = Flask(__name__)

@app.route('/')

def index():
  html = ''
  html += '<p>Hello World!</p>';
  #texts, tweets = search.twittersearch(q = ['War'], count = 10)
  
  #reviews, hashtable = twittertrend.trend(number = 5, lag = 0)

  result = stream.parsetw(track = 'Trump', number = 2)

  i = 0
  j = 0
  
  #for text in texts:
  #  html += 'text ' + str(i) + ': ' + text + '<br>'
  #  i += 1
  
  #for review in reviews:
  #  for r in review[0]:
  #    html += 'top ' + str(i) + ' review ' + str(j) + ' : ' + r + '<br>'
  #    j += 1
  #  i += 1

  for r in result[0]:
    html += 'result ' + str(i) + ' : ' + r
    i += 1
  
  html += '<p>' + sc.predict(movie_reviews.raw('pos/cv008_29435.txt')) + '</p>'
  return html
	# return "<p>Hello World!</p>" + "<p>" + sc.predict(movie_reviews.raw('pos/cv008_29435.txt')) + "</p>"
# return "<p>Hello World!</p>" + "<p>" + sa.sentiment(movie_reviews.raw('pos/cv008_29435.txt')) + "</p>"
##    return "<p>Hello World!</p>"

if __name__ == '__main__':
  app.run(port=8455, host = '140.116.177.150', debug=True)
