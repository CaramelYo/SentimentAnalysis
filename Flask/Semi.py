import nltk
import random
import pickle
import numpy as np
import gc
import warnings

from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.decomposition import LatentDirichletAllocation as LDA

##global info
vocabulary = []
classifiers = []
stopWords = stopwords.words('english')
wordTypes = ['J']
threshold = 0.7

MAX_ITER = 1500
DATA_NUM = 1900

classifierNames = ["MNB",
                   "BNB",
                   "LogisticRegression",
                   "SGD",
                   "LinearSVC",
                   "NuSVC",
                   "Voting"]

##to define the style. 0 => Normal style. Load documents, train, and save classifier. 1 => Module style. Load vote classifier 
style = 0
##to define the mode of select word. 0 => use word types. 1 => no word types
selectingWordsMode = 0
##to define whether the vocabulary need to be created. 0 => no. 1 => yes
isSetVocabulary = 1

def loadData():
    print('loading the dataset')
    
    dataset = [[word_tokenize(movie_reviews.raw(fileid)), category]
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    
    print('loading is  completed')
    return dataset

def wordsFilter(data):
##  getting the "if selectingWordsMode" out could improve the performance
    newWords = []

    if selectingWordsMode == 0:
        pos = nltk.pos_tag(data[0])
        for w in pos:
            if w[1][0] in wordTypes and w[0] not in stopWords and w[0] not in newWords:
                newWords.append(w[0])
    elif selectingWordsMode == 1:
        for word in words:
            if word not in stopWords and word not in newWords:
                newWords.append(word)
    
    data[0] = newWords
    del newWords

def wordsFilterWithVocabulary(data):
##  getting the "if selectingWordsMode" out could improve the performance
    newWords = []

    if selectingWordsMode == 0:
        pos = nltk.pos_tag(data[0])
        for w in pos:
            if w[1][0] in wordTypes and w[0] not in stopWords and w[0] not in newWords:
                newWords.append(w[0])
                if w[0] not in vocabulary:
                    vocabulary.append(w[0])
    elif selectingWordsMode == 1:
        for word in words:
            if word not in stopWords and word not in newWords:
                newWords.append(word)
                if word not in vocabulary:
                    vocabulary.append(word)

    data[0] = newWords
    del newWords

def buildLDAFeatures(zeros, words):
    global vocabulary

    features = zeros[:]
    for word in words:
        features[vocabulary.index(word)] += 1
    
    return features

def buildClassifierFeatures(words):
##    features = {}
    features = []
    for v in vocabulary:
        features.append(v in words)
##        features[v] = (v in words)
    
    return features

def buildTargetedClassifierFeatures(targetedData):
    trainingClassifierFeatures = []
    trainingTarget = []
    for data in targetedData:
        trainingClassifierFeatures.append(buildClassifierFeatures(data[0]))
        trainingTarget.append(data[1])

    trainingClassifierFeatures = np.array(trainingClassifierFeatures)
    trainingTarget = np.array(trainingTarget)
    
    return trainingClassifierFeatures, trainingTarget

def save(data, fileName):
    sf = open(fileName + '.pickle', 'wb')
    pickle.dump(data, sf)
    sf.close()

def load(fileName):
    lf = open(fileName + '.pickle', 'rb')
    data = pickle.load(lf)
    lf.close()
    return data

def accuracy(classifier, features, targets):
    number = len(targets)

    predictions = classifier.predict(features)

    correct = 0
    for i in range(number):
        if predictions[i] == targets[i]:
            correct += 1

    return correct / number

def mode(array):
    return max(array, key = array.count)

##to define the vote classifier
class VoteClassifier(ClassifierI):
    ##ctor
    ##*classifiers => ac, bc, cc, dc...    
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, feature):
##      *********** warning when feature is single
        votes = []
        for c in self._classifiers:
            v = c.predict(feature)
            votes.append(v[0])
        ##mode(): return the most common data
        ##when the numbers of two data are the same, get error
            
        return mode(votes)

    def confidence(self, feature):
##      *********** warning when feature is single
        votes = []
        for c in self._classifiers:
            v = c.predict(feature)
            votes.append(v[0])
        
        choiceVotes = votes.count(mode(votes))
        conf = choiceVotes / len(votes)
        return conf

    def predict(self, features):
        predictions = []
        for feature in features:
            predictions.append(self.classify(feature))
##            print('confidence = ', self.confidence(feature))

        return predictions

def main():
##  to disable the warnings
    warnings.filterwarnings("ignore")
    
    if style == 0:
        dataset = loadData()
        random.shuffle(dataset)

##      to get the part of dataset
##        dataset = dataset[:100]

##      to filter data
        print('filtering data start')
        
        if isSetVocabulary == 1:
            for data in dataset:
                wordsFilterWithVocabulary(data)
                
            save(vocabulary, 'Vocabulary')
            print('building vocabulary is completed\nvocabulary size =', len(vocabulary))
        elif isSetVocabulary == 0:
            for data in dataset:
                wordsFilter(data)
        
        print('filtering data is completed')
        
##        global vocabulary
##        save(vocabulary, 'Vocabulary')
##        print('vocabulary size =', len(vocabulary))

##        global vocabulary
##        vocabulary = load('Vocabulary')

        trainingData = dataset[:DATA_NUM]
        testData = dataset[DATA_NUM:]
        
        del dataset

        print('training targeted classifier features start')
        trainingClassifierFeatures, trainingTarget = buildTargetedClassifierFeatures(trainingData)
##        print(trainingTargetedClassifierFeatures[0])
##        save(trainingTargetedClassifierFeatures, 'trainingTargetedClassifierFeatures')
        print('training targeted classifier features are completed')

##      to train the sentiment models

        global classifiers
        print('classifiers training start!')
        gc.collect()
        
        
##      to use the naive bayes classifier to train. [ [{}, ''] , ....]
##        classifiers.append(nltk.NaiveBayesClassifier.train(trainingTargetedClassifierFeatures))
##        classifiers.append(

##    to try other classifier
##    MultinomialNB
##        classifiers.append(SklearnClassifier(MultinomialNB()))
        classifiers.append(MultinomialNB())

##    BernoulliNB
##        classifiers.append(SklearnClassifier(BernoulliNB()))
        classifiers.append(BernoulliNB())
##    LogisticRegression
##        classifiers.append(SklearnClassifier(LogisticRegression()))
        classifiers.append(LogisticRegression())

##    SGDClassifier
##        classifiers.append(SklearnClassifier(SGDClassifier()))
        classifiers.append(SGDClassifier())

##    SVC abort
##        classifiers.append(SklearnClassifier(SVC()))

##    LinearSVC
##        classifiers.append(SklearnClassifier(LinearSVC()))
        classifiers.append(LinearSVC())

##    NuSVC
##        classifiers.append(SklearnClassifier(NuSVC()))
        classifiers.append(NuSVC())

##    to train the classifier
        classifierNumber = len(classifiers)
        print('used classifier number = ', classifierNumber)

##    except naive bayes classifier
        for i in range(classifierNumber):
            print('i = ', i)
            classifiers[i].fit(trainingClassifierFeatures, trainingTarget)

##    to use our vote classifier
        voteClassifier = VoteClassifier(classifiers[0],
                                        classifiers[1],
                                        classifiers[2],
                                        classifiers[3],
                                        classifiers[4],
                                        classifiers[5])
        classifiers.append(voteClassifier)

        print('classifiers training end!')

##      to predict
##        testClassifierFeatures = [(buildClassifierFeatures(data[0])) for data in testData]
##        save(testClassifierFeatures, 'testClassifierFeatures')
##        print('test classifier features are completed')
        
##        testTargetedClassifierFeatures = [buildTargetedClassifierFeatures(data) for data in testData]
##        save(testTargetedClassifierFeatures, 'testTargetedClassifierFeatures')
        testClassifierFeatures, testTarget = buildTargetedClassifierFeatures(testData)
        print('test targeted classifier features are completed')
        
        print('predicting start')

##        for i in range(len(classifiers)):
##            print(classifierNames[i], 'Algo accuracy percent: ', (nltk.classify.accuracy(classifiers[i], testTargetedClassifierFeatures)) * 100)
        for i in range(len(classifiers)):
            print(classifierNames[i], 'Algo accuracy percent: ', (accuracy(classifiers[i], testClassifierFeatures, testTarget)) * 100)

##        for test in testTargetedClassifierFeatures:
##            print("Classification: ", voteClassifier.classify(test[0]), " Confidence: ", voteClassifier.confidence(test[0]), 'in real: ', test[1])

##      to save the classifier as pickle
        for i in range(classifierNumber):
            save(classifiers[i], classifierNames[i])
        
##        predictions = []
##
##        for feature in testClassifierFeatures:
##            predictions.append([voteClassifier.classify(feature), voteClassifier.confidence(feature)])
##
##        for prediction in predictions:
##            print("Classification: ", prediction[0], " Confidence: ", prediction[1])

####      to add to training targetd classifier features if confidence > threshold
##        newTestClassifierFeatures = []
##        newTestData = []
##        for i in range(len(predictions)):
##            if predictions[i][1] > threshold:
##                trainingData.append(testData[i])
##                trainingTargetedClassifierFeatures.append([testClassifierFeatures[i], predictions[i][0]])
##            else:
##                newTestData.append(testData[i])
##                newTestClassifierFeatures.append(testClassifierFeatures[i])
##
##        testData = newTestData
##        testClassifierFeatures = newTestClassifierFeatures
        
####      to train the LDA
##        zeros = [0 for n in range(len(vocabulary))]
##        trainingLDAFeatures = [(buildLDAFeatures(zeros, data[0])) for data in trainingData]
##
##        print('LDA start')
##        model = LDA(n_topics = 2, max_iter = 1500, learning_method = 'online')
##        topicDistributions = model.fit_transform(trainingLDAFeatures)
##        print('LDA end')
##
##        save(model, 'LDAModel')
##
##        for distribution in topicDistributions:
##            print(distribution)
##
##        
##
##        
##
##        testFeatures = [buildFeatures(zeros, i, testData[i][0]) for i in range(len(testData))]
##        print(model.transform(testFeatures))
##
##        print('ground truth')
##        for data in testData:
##            print(data[1])
##
####        topic = model.components_
####        print(topic)
##
##        n_top_words = 8
##        for i, topic_dist in enumerate(model.components_):
##            topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
##            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    elif style == 1:
        print('YO')
        
main()
