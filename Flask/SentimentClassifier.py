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
wordTypes = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
trainingRatio = 0.9
trainingSize = 500
threshold = 0.7

# classifierNames = ["MNB",
#                    "BNB",
#                    "LogisticRegression",
#                    "SGD",
#                    "LinearSVC",
#                    "NuSVC",
#                    "Voting"]

classifierNames = ["MNB",
                   "BNB",
                   "SGD",
                   "Voting"]

##to define the style. 0 => Normal style. Load documents, train, and save classifier. 1 => Module style. Load vote classifier 
style = 0
##to define the mode of select word. 0 => use word types. 1 => no word types
# selectingWordsMode = 0
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

    pos = nltk.pos_tag(data[0])
    for w in pos:
        if w[1] in wordTypes and w[0] not in stopWords and w[0] not in newWords:
            newWords.append(w[0])

    data[0] = newWords
    del newWords

    # if selectingWordsMode == 0:
    #     pos = nltk.pos_tag(data[0])
    #     for w in pos:
    #         if w[1] in wordTypes and w[0] not in stopWords and w[0] not in newWords:
    #             newWords.append(w[0])
    # elif selectingWordsMode == 1:
    #     for word in words:
    #         if word not in stopWords and word not in newWords:
    #             newWords.append(word)
    
    # data[0] = newWords
    # del newWords

def wordsFilterWithVocabulary(data):
##  getting the "if selectingWordsMode" out could improve the performance
    newWords = []

    pos = nltk.pos_tag(data[0])
    for w in pos:
        if w[1] in wordTypes and w[0] not in stopWords and w[0] not in newWords:
            newWords.append(w[0])
            if w[0] not in vocabulary:
                vocabulary.append(w[0])

    data[0] = newWords
    del newWords

    # if selectingWordsMode == 0:
    #     pos = nltk.pos_tag(data[0])
    #     for w in pos:
    #         if w[1] in wordTypes and w[0] not in stopWords and w[0] not in newWords:
    #             newWords.append(w[0])
    #             if w[0] not in vocabulary:
    #                 vocabulary.append(w[0])
    # elif selectingWordsMode == 1:
    #     for word in words:
    #         if word not in stopWords and word not in newWords:
    #             newWords.append(word)
    #             if word not in vocabulary:
    #                 vocabulary.append(word)

def buildClassifierFeatures(words):
    features = []
    for v in vocabulary:
        features.append(v in words)
    
    return features

def buildTargetedClassifierFeatures(targetedData):
    classifierFeatures = []
    targets = []
    for data in targetedData:
        classifierFeatures.append(buildClassifierFeatures(data[0]))
        targets.append(data[1])

    # to change type to np
    classifierFeatures = np.array(classifierFeatures)
    targets = np.array(targets)

    # print(classifierFeatures.shape)
    # print(targets.shape)
    
    return classifierFeatures, targets

def writeToTxt(data, fileName):
    with open(fileName + '.txt', 'w') as file:
        for word in data:
            print(word, file = file)

def save(data, fileName):
    with open(fileName + '.pickle', 'wb') as file:
        pickle.dump(data, file)

def load(fileName):
    with open(fileName + '.pickle', 'rb') as file:
        data = pickle.load(file)
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
            # why v[0]
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

        # # to get the part of dataset
        # dataset = dataset[:1000]

        # to filter data
        print('filtering data start')
        if isSetVocabulary == 1:
            for data in dataset:
                wordsFilterWithVocabulary(data)
                
            save(vocabulary, 'Vocabulary')
            print('building vocabulary is completed and saved\nvocabulary size =', len(vocabulary))
        elif isSetVocabulary == 0:
            for data in dataset:
                wordsFilter(data)

        print('filtering data is completed')

        print('saving filtered dataset start')
        save(dataset, 'FilteredDataset')
        print('saving filtered dataset end')

        # # to get data from pickle
        # global vocabulary
        # dataset = load('FilteredDataset')
        # vocabulary = load('Vocabulary')

        # to split dataset into two part, training data and test data
        totalSize = len(dataset)
        print('total size = ', totalSize)
        trainingDataSize = int(totalSize * trainingRatio)
        testDataSize = totalSize - trainingDataSize
        print('training data size = ', trainingDataSize)
        print('test data size = ', testDataSize)
        trainingData = dataset[:trainingDataSize]
        testData = dataset[trainingDataSize:]

        del dataset

        print('training targeted classifier features start')
        trainingClassifierFeatures, trainingTargets = buildTargetedClassifierFeatures(trainingData)
        print('training targeted classifier features are completed')

        print('saving training classifier features and training targets start')
        save(trainingClassifierFeatures, 'TrainingClassifierFeatures')
        save(trainingTargets, 'TrainingTargets')
        print('saving training classifier features and training targets end')

        # # to get data from pickle
        # trainingClassifierFeatures = load('TrainingClassifierFeatures')
        # trainingTargets = load('TrainingTargets')

        # to train the sentiment models
        global classifiers
        print('classifiers creating start')
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
##    LogisticRegression abort because it doesn't have 'partial_fit'
##        classifiers.append(SklearnClassifier(LogisticRegression()))
        # classifiers.append(LogisticRegression())

##    SGDClassifier
##        classifiers.append(SklearnClassifier(SGDClassifier()))
        classifiers.append(SGDClassifier())

##    SVC abort
##        classifiers.append(SklearnClassifier(SVC()))

##    LinearSVC
##        classifiers.append(SklearnClassifier(LinearSVC()))
        # classifiers.append(LinearSVC())

##    NuSVC
##        classifiers.append(SklearnClassifier(NuSVC()))
        # classifiers.append(NuSVC())

        print('classifiers creating end')

        # to train the classifier
        classifierNumber = len(classifiers)
        print('used classifier number = ', classifierNumber)

# ##    except naive bayes classifier
#         for i in range(classifierNumber):
#             # print('i = ', i)
#             classifiers[i].fit(trainingClassifierFeatures, trainingTarget)

        print('classifiers partial training start')
        start = 0
        end = 0
        tempSize = trainingDataSize
        # it is passed to partial_fit when first call
        allTargets = np.array(['neg', 'pos'])

        # first call
        if(tempSize > trainingSize):
            end += trainingSize
            for i in range(classifierNumber):
                # print('i = ', i)
                classifiers[i].partial_fit(trainingClassifierFeatures[start: end], trainingTargets[start: end], classes = allTargets)

            tempSize -= trainingSize
            start = end

        # to partially train continually
        while(tempSize > trainingSize):
            end += trainingSize
            for i in range(classifierNumber):
                classifiers[i].partial_fit(trainingClassifierFeatures[start: end], trainingTargets[start: end])

            tempSize -= trainingSize
            start = end

        # last call
        end += tempSize
        for i in range(classifierNumber):
            classifiers[i].partial_fit(trainingClassifierFeatures[start: end], trainingTargets[start: end])

        # to use our vote classifier
        # can use function like 'add' to add classifier?
        voteClassifier = VoteClassifier(classifiers[0],
                                        classifiers[1],
                                        classifiers[2])
        classifiers.append(voteClassifier)

        print('classifiers partial training start')

        print('test targeted classifier features start')
        testClassifierFeatures, testTargets = buildTargetedClassifierFeatures(testData)
        print('test targeted classifier features are completed')

        print('saving test classifier features and test targets start')
        save(testClassifierFeatures, 'TestClassifierFeatures')
        save(testTargets, 'TestTargets')
        print('saving test classifier features and test targets end')
        
        # to predict
        print('predicting start')

        for i in range(len(classifiers)):
            print(classifierNames[i], 'Algo accuracy percent: ', (accuracy(classifiers[i], testClassifierFeatures, testTargets)) * 100)    
            # print("Classification: ", voteClassifier.classify(test[0]), " Confidence: ", voteClassifier.confidence(test[0]), 'in real: ', test[1])

        # to save the classifier as pickle except vote classifier
        temp = classifierNumber - 1
        for i in range(temp):
            save(classifiers[i], classifierNames[i])
        
        # # to add good test data into training data
        # predictions = []
        # for feature in testClassifierFeatures:
        #     predictions.append([voteClassifier.classify(feature), voteClassifier.confidence(feature)])

        # for prediction in predictions:
        #     print("Classification: ", prediction[0], " Confidence: ", prediction[1])

        # # to add to training targetd classifier features if confidence > threshold
        # # newTestClassifierFeatures = []
        # # newTestData = []
        # for i in range(len(predictions)):
        #     if predictions[i][1] > threshold:
        #         trainingData.append(testData[i])
        #         trainingClassifierFeatures.append(testClassifierFeatures[i])
        #         trainingTargets.append(predictions[i][0])

        #         testData.remove(testData[i])
        #         testClassifierFeatures.remove(testClassifierFeatures[i])
        #     # else:
        #     #     newTestData.append(testData[i])
        #     #     newTestClassifierFeatures.append(testClassifierFeatures[i])

        # # testData = newTestData
        # # testClassifierFeatures = newTestClassifierFeatures
    elif style == 1:
        print('YO')
        
main()