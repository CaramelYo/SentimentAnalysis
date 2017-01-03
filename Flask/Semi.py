import nltk
import random
import pickle
import numpy as np

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
DATA_NUM = 1990

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

def dataFilter(data):
##  getting the "if selectingWordsMode" out could improve the performance
    newData = []

    if selectingWordsMode == 0:
        pos = nltk.pos_tag(data[0])
        for w in pos:
            if w[1][0] in wordTypes and w[0] not in stopWords and w[0] not in newData:
                newData.append(w[0])
                if isSetVocabulary == 1 and w[0] not in vocabulary:
                    vocabulary.append(w[0])
    elif selectingWordsMode == 1:
        for word in data[0]:
            if word not in stopWords and word not in newData:
                newData.append(word)
                if isSetVocabulary == 1 and word not in vocabulary:
                    vocabulary.append(word)

    return newData

def buildLDAFeatures(zeros, words):
    global vocabulary

    features = zeros[:]
    for word in words:
        features[vocabulary.index(word)] += 1
    
    return features

def buildClassifierFeatures(words):
    global vocabulary
    
    features = {}
    for v in vocabulary:
        features[v] = (v in words)
    
    return features

def buildTargetedClassifierFeatures(words, target):
    classifierFeatures = buildClassifierFeatures(words)
    targetedClassifierFeatures = [classifierFeatures, target]

    return targetedClassifierFeatures

def save(data, fileName):
    sf = open(fileName + '.pickle', 'wb')
    pickle.dump(data, sf)
    sf.close()

def load(fileName):
    lf = open(fileName + '.pickle', 'rb')
    data = pickle.load(lf)
    lf.close()
    return data

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

def main():
    if style == 0:
        dataset = loadData()
        random.shuffle(dataset)

##      to get the part of dataset
##        dataset = dataset[:DATA_NUM]

##      to filter data
        for data in dataset:
            data[0] = dataFilter(data)

        if isSetVocabulary == 1:
            save(vocabulary, 'Vocabulary')
            print('building vocabulary is completed\nvocabulary size =', len(vocabulary))

        print('filtering data is completed')
        
##        global vocabulary
##        save(vocabulary, 'Vocabulary')
##        print('vocabulary size =', len(vocabulary))

##        global vocabulary
##        vocabulary = load('Vocabulary')

        trainingData = dataset[:DATA_NUM]
     
        trainingTargetedClassifierFeatures = [(buildTargetedClassifierFeatures(data[0], data[1])) for data in trainingData]
        save(trainingTargetedClassifierFeatures, 'trainingTargetedClassifierFeatures')
        print('training targeted classifier features are completed')

##      to train the sentiment models

        global classifiers
        print('classifiers training start!')
        
##      to use the naive bayes classifier to train. [ [{}, ''] , ....]
        classifiers.append(nltk.NaiveBayesClassifier.train(trainingTargetedClassifierFeatures))

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
##    print(length)

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

        print('classifiers training end!')

##      to predict
        testData = dataset[DATA_NUM:]
        testClassifierFeatures = [(buildClassifierFeatures(data[0])) for data in testData]
        save(testClassifierFeatures, 'testClassifierFeatures')
        print('test classifier features are completed')

        print('predicting start')
        predictions = []

        for feature in testClassifierFeatures:
            predictions.append([voteClassifier.classify(feature), voteClassifier.confidence(feature)])

        for prediction in predictions:
            print("Classification: ", prediction[0], " Confidence: ", prediction[1])


##      to add to training targetd classifier features if confidence > threshold
        newTestClassifierFeatures = []
        newTestData = []
        for i in range(len(predictions)):
            if predictions[i][1] > threshold:
                trainingData.append(testData[i])
                trainingTargetedClassifierFeatures.append([testClassifierFeatures[i], predictions[i][0]])
            else:
                newTestData.append(testData[i])
                newTestClassifierFeatures.append(testClassifierFeatures[i])

        testData = newTestData
        testClassifierFeatures = newTestClassifierFeatures
        

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
##    elif style == 1:
##        print('YO')
        
main()
