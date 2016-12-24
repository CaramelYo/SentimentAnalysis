##import numpy as np
import nltk
import random
import pickle
import numpy as np
##import lda
##import lda.datasets

from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.tokenize import word_tokenize

##X = lda.datasets.load_reuters()
##print (X)
##vocab = lda.datasets.load_reuters_vocab()
##print (len(vocab))
####print(vocab)
##print(type(vocab))
##print(vocab[1])
####titles = lda.datasets.load_reuters_titles()
####print (titles)
##print (X.shape)
####print (X.sum())
####
####model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
####model.fit(X)  # model.fit_transform(X) is also available
####topic_word = model.topic_word_  # model.components_ also works
####n_top_words = 8
####for i, topic_dist in enumerate(topic_word):
####    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
####    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

vocabulary = []
stopWords = stopwords.words('english')
wordTypes = ['J']

MAX_ITER = 1500
DATA_NUM = 1990

##to define the style. 0 => Normal style. Load documents, train, and save classifier. 1 => Module style. Load vote classifier 
style = 0

def loadData():
    print('loading the dataset')
    
    dataset = [[word_tokenize(movie_reviews.raw(fileid)), category]
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    
    print('loading is  completed')
    return dataset

def buildVocabulary(dataset):
    global vocabulary

    for data in dataset:
        pos = nltk.pos_tag(data[0])
        newData = []
        for w in pos:
            if w[1][0] in wordTypes and w[0] not in stopWords and w[0] not in newData:
                newData.append(w[0])
                if w[0] not in vocabulary:
                    vocabulary.append(w[0])

        data[0] = newData
        
    print('building vocabulary is completed')

def buildFeatures(zeros, i, words):
    global vocabulary

    features = zeros[:]
    for word in words:
        if word not in stopWords:
            if word in vocabulary:
                features[vocabulary.index(word)] += 1
            else:
                print('GG')
    
    return features

def save(data, fileName):
    sf = open(fileName + '.pickle', 'wb')
    pickle.dump(data, sf)
    sf.close()

def load(fileName):
    lf = open(fileName + '.pickle', 'rb')
    data = pickle.load(lf)
    lf.close()
    return data    

def main():
    if style == 0:
        dataset = loadData()
        random.shuffle(dataset)

##        dataset = dataset[:DATA_NUM]

        buildVocabulary(dataset)
        global vocabulary
        save(vocabulary, 'Vocabulary')
        print('vocabulary size =', len(vocabulary))
##        global vocabulary
##        vocabulary = load('Vocabulary')

        trainingData = dataset[:DATA_NUM]
        testData = dataset[DATA_NUM:]
        
        zeros = [0 for n in range(len(vocabulary))]
        trainingFeatures = [(buildFeatures(zeros, i, trainingData[i][0])) for i in range(len(trainingData))]

        print('LDA start')
        model = LDA(n_topics = 2, max_iter = 1500, learning_method = 'online')
        model.fit(trainingFeatures)
        print('LDA end')

        save(model, 'LDAmodel')

        testFeatures = [buildFeatures(zeros, i, testData[i][0]) for i in range(len(testData))]
        print(model.transform(testFeatures))

        print('ground truth')
        for data in testData:
            print(data[1])

##        topic = model.components_
##        print(topic)

        n_top_words = 8
        for i, topic_dist in enumerate(model.components_):
            topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    elif style == 1:
        print('YO')
        
main()
