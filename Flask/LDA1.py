import nltk
import random
from nltk.corpus import movie_reviews
import pickle
import operator

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from nltk.tokenize import word_tokenize

vocabulary = {}

FEATURE_NUM = 2000
CV_ITERS_NUM = 50
TRAINING_NUM = 1500

##to define the style. 0 => Normal style. Load documents, train, and save classifier. 1 => Module style. Load vote classifier 
style = 0

def loadData():
    print('loading the dataset')
    
    dataset = [(list(word_tokenize(movie_reviews.raw(fileid))), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    
    print('loading is  completed')
    return dataset

def buildVocabulary(dataset):
    print('building vocabulary of words in the dataset')
    
##  to get the global variable
    global vocabulary

    for data in dataset:
        for word in data[0]:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1

    print('building is completed')
    return vocabulary
    
def buildFeatures(words, mostWords):
    features = []

    for i in range(len(mostWords)):
        features.append(mostWords[i] in words)
    return features


def extractFeatures(dataset):
    vocabulary = buildVocabulary(dataset)

    global FEATURE_NUM

##  to get the most common words from vocabulary  

##    words = dict(sorted(vocabulary.items(), key = lambda d: d[1], reverse = True)[:FEATURE_NUM])
##    words = words.keys()

    items = vocabulary.items()
    backItems = [[v[1], v[0]] for v in items]
    backItems.sort(reverse = True)
    words = [backItems[i][1] for i in range(FEATURE_NUM)]
    print(vocabulary[words[0]])
    print(vocabulary[words[1]])

    print('extracting features')
    
    features = [(buildFeatures(dataset[i][0], words)) for i in range(len(dataset))]

    print('extracting features is completed')
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

def trainingModel(features, targets, tol = None):
    print('training model')

    if tol is not None:
        classifier = LDA(tol = tol)
    else:
        classifier = LDA()
    
    classifier.fit(features, targets)

    print('training model is completed')
    return classifier
    
##classifier = LDA()

##print (word_tokenize(movie_reviews.raw('pos/cv000_29590.txt')))


def main():
    if style == 0:
        dataset = loadData()
        random.shuffle(dataset)

        features = extractFeatures(dataset)
        save(features, 'Features')

        trainingFeatures = features[:TRAINING_NUM]
        trainingTargets = []
        for i in range(TRAINING_NUM):
            trainingTargets.append(dataset[i][1])
        
        testFeatures = features[TRAINING_NUM:]
        testTargets = []
        for i in range(TRAINING_NUM, len(dataset)):
            testTargets.append(dataset[i][1])

        classifier = trainingModel(trainingFeatures, trainingTargets)
        save(classifier, 'LDA1')

        score = classifier.score(testFeatures, testTargets)
        print('Accuracy against test set is {} percent'.format(score * 100))
    elif style == 1:
        features = load('Features')
        classifier = load('LDA1')
        
        testFeatures = features[TRAINING_NUM:]
        testTargets = []
        for i in range(TRAINING_NUM, len(dataset)):
            testTargets.append(dataset[i][1])

        score = classifier.score(testFeatures, testTargets)
        print('Accuracy against test set is {} percent'.format(score * 100))        
        
        
main()
