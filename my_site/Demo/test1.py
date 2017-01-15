import tweepysearch as search
import string
from collections import Counter
from nltk.corpus import stopwords
from bokeh.charts import output_file, show, Bar

def plotchart(q):
  text, time = search.twittersearch(q = q, number = 10)
  
  punctuation = list(string.punctuation)
  stop = stopwords.words('english') + punctuation + ['RT', 's', 'u2026', 'The', 'amp', 'By', 'points',
        'K', 'You', 'I', 'said', 'https', 'u2019s', '8', 'one', 'want','via', 'This', 'As', 'says', 'would', 'next',
        'why', 'u2019', 'u2605', 'A', 'htt', 'ud83d', 'like', '1', 'u2013']

  count_all = Counter()

  for i in range(0, len(text), 1):
    terms_stop = [term for term in preprocess(text[i]) if term not in stop]
    # Update the counter
    count_all.update(terms_stop)

  words = pd.DataFrame.from_dict(count_all.most_common(20), orient='columns', dtype = None)
  words.columns = ['Key Words', 'Words Count']

  a = 'Words Distribution of '
  title = a + q
  bar = Bar(words, 'Key Words',values='Words Count',
            title = title, legend='top_right', bar_width=0.5)
  output_file("bar_search.html", title="bar_search.py example")
  show(bar)

plotchart('Trump')
