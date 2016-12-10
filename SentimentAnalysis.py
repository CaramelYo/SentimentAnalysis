import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

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
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choiceVotes = votes.count(mode(votes))
        conf = choiceVotes / len(votes)
        return conf
        

##to load the movie reviews text
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

##to random the order in movie reviews
random.shuffle(documents)

##print(documents[1])

##to read all the words in allWords
allWords = []
for w in movie_reviews.words():
    allWords.append(w.lower())

##to calculate the freq of words
wordsFreq = nltk.FreqDist(allWords)

##print(wordsFreq.most_common(15))
##
##print(wordsFreq["beautiful"])

##print(wordsFreq["beautiful"])

##to create the words features
wordsFeatures = list(wordsFreq.keys())[:30]

##print(wordsFeatures[0])

def findFeatures(document):
    ##to let the list loop
    words = set(document)
    features = {}
    ##to find the features in words (hava => true)
    for w in wordsFeatures:
        features[w] = (w in words)

    return features

##to get all features in documents
##print((findFeatures(movie_reviews.words('pos/cv000_29590.txt'))))
featureSets = [(findFeatures(rev), category) for (rev, category) in documents]

print(featureSets[0])

##to set the training set and test set
trainingSet = featureSets[:1500]
testSet = featureSets[1500:]

##to use the naive bayes classifier to train
classifier = nltk.NaiveBayesClassifier.train(trainingSet)

##to use the saved pickle rather than training
##fClassifier = open("NaiveBayes.pickle", "rb")
##loadedClassifier = pickle.load(fClassifier)
##fClassifier.close()

##to use the naive bayes classifier to test
print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testSet)) * 100)
##print("Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(loadedClassifier, testSet)) * 100)
##classifier.show_most_informative_features(15)

##to save the classifier as pickle
##savedClassifier = open("NaiveBayes.pickle", "wb")
##pickle.dump(classifier, savedClassifier)
##savedClassifier.close()

##to try other classifier
##MultinomialNB
MNBClassifier = SklearnClassifier(MultinomialNB())
MNBClassifier.train(trainingSet)
print("MNB classifier accuracy percent: ", (nltk.classify.accuracy(MNBClassifier, testSet)) * 100)

##GaussianNB
##GNBClassifier = SklearnClassifier(GaussianNB())
##GNBClassifier.train(trainingSet)
##print("GNB classifier accuracy percent: ", (nltk.classify.accuracy(GNBClassifier, testSet)) * 100)

##BernoulliNB
BNBClassifier = SklearnClassifier(BernoulliNB())
BNBClassifier.train(trainingSet)
print("BNB classifier accuracy percent: ", (nltk.classify.accuracy(BNBClassifier, testSet)) * 100)

##LogisticRegression
LogisticRegressionClassifier = SklearnClassifier(LogisticRegression())
LogisticRegressionClassifier.train(trainingSet)
print("LogisticRegression classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegressionClassifier, testSet)) * 100)

##SGDClassifier
SGDClassifier = SklearnClassifier(SGDClassifier())
SGDClassifier.train(trainingSet)
print("SGD classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier, testSet)) * 100)

##SVC
SVCClassifier = SklearnClassifier(SVC())
SVCClassifier.train(trainingSet)
print("SVC classifier accuracy percent: ", (nltk.classify.accuracy(SVCClassifier, testSet)) * 100)

##LinearSVC
LinearSVCClassifier = SklearnClassifier(LinearSVC())
LinearSVCClassifier.train(trainingSet)
print("LinearSVC classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVCClassifier, testSet)) * 100)

##NuSVC
NuSVCClassifier = SklearnClassifier(NuSVC())
NuSVCClassifier.train(trainingSet)
print("NuSVC classifier accuracy percent: ", (nltk.classify.accuracy(NuSVCClassifier, testSet)) * 100)


##to use our vote classifier
voteClassifier = VoteClassifier(classifier,
                                MNBClassifier,
                                BNBClassifier,
                                LogisticRegressionClassifier,
                                SGDClassifier,
                                SVCClassifier,
                                LinearSVCClassifier,
                                NuSVCClassifier)

print("vote classifier accuracy percent: ", (nltk.classify.accuracy(voteClassifier, testSet)) * 100)

print("Classification: ", voteClassifier.classify(testSet[0][0]), " Confidence: ", voteClassifier.confidence(testSet[0][0]))
print("Classification: ", voteClassifier.classify(testSet[1][0]), " Confidence: ", voteClassifier.confidence(testSet[1][0]))
print("Classification: ", voteClassifier.classify(testSet[2][0]), " Confidence: ", voteClassifier.confidence(testSet[2][0]))
print("Classification: ", voteClassifier.classify(testSet[3][0]), " Confidence: ", voteClassifier.confidence(testSet[3][0]))
print("Classification: ", voteClassifier.classify(testSet[4][0]), " Confidence: ", voteClassifier.confidence(testSet[4][0]))
print("Classification: ", voteClassifier.classify(testSet[5][0]), " Confidence: ", voteClassifier.confidence(testSet[5][0]))
                
                                
