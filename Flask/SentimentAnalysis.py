import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
##from statistics import mode
from nltk.tokenize import word_tokenize

##def mode(array):
##    most = (max(list(map(array.count, array))))
##    return list(set(filter(lambda x: array.count(x) == most, array)))

DATA_NUM = 1900

def mode(array):
    return max(array, key = array.count)

##to define the vote classifier
class VoteClassifier(ClassifierI):
    ##ctor
    ##*classifiers => ac, bc, cc, dc...    
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        ##mode(): return the most common data
        ##when the numbers of two data are the same, get error

##        print(mode(votes))
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

##        print(mode(votes))
        choiceVotes = votes.count(mode(votes))
        conf = choiceVotes / len(votes)
        return conf

    def accuracy(testSet):
        num = len(testSet)
        correct = 0

        for test in testSet:
            cla
        
        return
    
def findFeatures(document):
    ##to let the list loop
    words = set(document)
    features = {}
    ##to find the features in words (hava => true)
    for w in wordFeatures:
        features[w] = (w in words)

    return features

def findFeaturesInSentance(sentance):
    words = word_tokenize(sentance)
    features = {}
    for w in wordFeatures:
        features[w] = (w in words)
        
    return features
    
def sentiment(text):
    print("YO")
    feats = findFeaturesInSentance(text)
##    print(feats)
    print(voteClassifier)
    return voteClassifier.classify(feats)

classifierNames = ["NaiveBayes",
                      "MNB",
                      "BNB",
                      "LogisticRegression",
                      "SGD",
                      "SVC",
                      "LinearSVC",
                      "NuSVC"]

classifiers = []

##to define the style. 0 => Normal style. Load documents, train, and save classifier. 1 => Module style. Load vote classifier 
style = 0

if style == 0:
##    to load the movie reviews text
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    print(documents[0])

##    to random the order in movie reviews
    random.shuffle(documents)

##    print(documents[1])

##    to read all the words in allWords
    allWords = []
    for w in movie_reviews.words():
        allWords.append(w.lower())

##    to calculate the freq of words
    wordsFreq = nltk.FreqDist(allWords)

##    print(wordsFreq.most_common(15))

##    print(wordsFreq["beautiful"])

##    to create the words features
    wordFeatures = list(wordsFreq.keys())[:5000]

##    print(wordFeatures[0])

##    to save word features
    sf = open("WordFeatures.pickle", "wb")
    pickle.dump(wordFeatures, sf)
    sf.close()    

##    to get all features in documents
##    print((findFeatures(movie_reviews.words('pos/cv000_29590.txt'))))
    featureSets = [(findFeatures(rev), category) for (rev, category) in documents]

##    print(featureSets[0])

##    to set the training set and test set
    trainingSet = featureSets[:DATA_NUM]
    testSet = featureSets[DATA_NUM:]

##    to use the naive bayes classifier to train
    classifiers.append(nltk.NaiveBayesClassifier.train(trainingSet))

##    to try other classifier
##    MultinomialNB
    classifiers.append(SklearnClassifier(MultinomialNB()))

##    BernoulliNB
    classifiers.append(SklearnClassifier(BernoulliNB()))

##    LogisticRegression
    classifiers.append(SklearnClassifier(LogisticRegression()))

##    SGDClassifier
    classifiers.append(SklearnClassifier(SGDClassifier()))

##    SVC
    classifiers.append(SklearnClassifier(SVC()))

##    LinearSVC
    classifiers.append(SklearnClassifier(LinearSVC()))

##    NuSVC
    classifiers.append(SklearnClassifier(NuSVC()))

##    to train the classifier
    length = len(classifiers)
    print('used classifier number = ', length)

##    except naive bayes classifier
    for i in range(1, length):
        classifiers[i].train(trainingSet)

##    to use our vote classifier
    voteClassifier = VoteClassifier(classifiers[0],
                                      classifiers[1],
                                      classifiers[2],
                                      classifiers[3],
                                      classifiers[4],
                                      classifiers[5],
                                      classifiers[6],
                                      classifiers[7])
    classifiers.append(voteClassifier)

##    to predict
    testNum = len(testSet)
    print('test number = ', testNum)
    
    length = len(classifiers)
    for i in range(0, length):
        print(classifierNames[i], "Algo accuracy percent: ", (nltk.classify.accuracy(classifiers[i], testSet)) * 100)

##    classifier.show_most_informative_features(15)##    print(testSet[0][0])

##    to predict by vote classifier
    
##    for i in range(0, testNum):
##        print("Classification: ", voteClassifier.classify(testSet[i][0]), " Confidence: ", voteClassifier.confidence(testSet[i][0]))

    print('Voting classifier accuracy percent = ', voteClassifier.accuracy(testSet) * 100)

##    to save the classifier as pickle
    length = len(classifierNames)
    for i in range(0, length):
        sf = open(classifierNames[i] + ".pickle", "wb")
        pickle.dump(classifiers[i], sf)
        sf.close()
    
elif style == 1:
##    to load word features
    lf = open("WordFeatures.pickle", "rb")
    wordFeatures = pickle.load(lf)
    lf.close()
    
##    to use the saved pickle rather than training
    length = len(classifierNames)
    for i in range(0, length):
        lf = open(classifierNames[i] + ".pickle", "rb")
        classifiers.append(pickle.load(lf))
        lf.close()

    voteClassifier = VoteClassifier(classifiers[0],
                                      classifiers[1],
                                      classifiers[2],
                                      classifiers[3],
                                      classifiers[4],
                                      classifiers[5],
                                      classifiers[6],
                                      classifiers[7])

##print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and python was the best."))

