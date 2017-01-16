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

import delete

##global info
vocabulary = []
classifiers = []
discourse = []
sentenceSpliter = ['.', ',', '--']
stopWords = stopwords.words('english')
wordTypes = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
trainingRatio = 0.8
threshold = 0.7

#to define training mode. 0 => partial training for MNB BNB SGD. 1 => fitting for LG. 2 => fitting for SVC LinearSVC NuSVC
trainingMode = 3

# classifierNames = ["MNB",
#                    "BNB",
#                    "LogisticRegression",
#                    "SGD",
#                    "LinearSVC",
#                    "NuSVC",
#                    "Voting"]

if(trainingMode == 0):
    #partial training size
    trainingSize = 500
elif(trainingMode == 1):
    trainingSize = 5500
elif(trainingMode == 2):
    trainingSize = 800

if(trainingMode == 0):
    classifierNames = ['MNB', 'BNB', 'SGD', 'Voting']
elif(trainingMode == 1):
    classifierNames = ['LogisticRegression', 'Voting']
elif(trainingMode == 2):
    classifierNames = ['SVC', 'LinearSVC', 'NuSVC', 'Voting']
elif(trainingMode == 3):
    classifierNames = ['MNB', 'BNB', 'SGD', 'LogisticRegression', 'SVC', 'LinearSVC', 'NuSVC', 'Voting']

dataFileNames = ['FilteredShortMovieReviews', 'FilteredMovieReviews']

##to define the style. 0 => Normal style. Load documents, train, and save classifier. 1 => Module style. Load vote classifier 
style = 0
##to define the mode of select word. 0 => use word types. 1 => no word types
# selectingWordsMode = 0
##to define whether the vocabulary need to be created. 0 => no. 1 => yes
isSetVocabulary = 1
#movie reviews or short reviews
isMovieReviews = 0

def loadData():
    print('loading the dataset')
    
    if(isMovieReviews == 1):
        dataset = [[word_tokenize(movie_reviews.raw(fileid)), category]
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]
    elif(isMovieReviews == 0):
        with open('Data/PositiveReviews.txt') as f:
            lines = f.readlines()

        dataset = [[word_tokenize(line), 'pos']
                  for line in lines]

        with open('Data/NegativeReviews.txt') as f:
            lines = f.readlines()

        dataset.extend([[word_tokenize(line), 'neg']
                        for line in lines])

    print('loading is  completed')
    return dataset

def loadDiscourse():
    print('loading discourse starts')

    discourse = []
    with open('Vocabulary for Sentiment Analysis/BackDiscourse.txt') as f:
        for line in f:
            discourse.append(word_tokenize(line))

    print('loading discourse ends')
    return discourse

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

def wordsFilterIntext(words):
##  getting the "if selectingWordsMode" out could improve the performance
    newWords = []

    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1] in wordTypes and w[0] not in stopWords and w[0] not in newWords:
            newWords.append(w[0])

    # data[0] = newWords
    # words = newWords
    # del newWords
    return newWords

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

def buildClassifierFeature(words):
    global vocabulary

    feature = []
    for v in vocabulary:
        feature.append(v in words)
    
    #feature = {}
    #for v in vocabulary:
    #    feature[v] = (v in words)

    return feature

def buildTargetedClassifierFeatures(targetedData):
    classifierFeatures = []
    targets = []
    for data in targetedData:
        classifierFeatures.append(buildClassifierFeature(data[0]))
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
    with open('Pickle/' + fileName + '.pickle', 'wb') as file:
        pickle.dump(data, file)

def load(fileName):
    with open('Pickle/' + fileName + '.pickle', 'rb') as file:
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

def predict(text):
    print('text preprocessing starts')
    print('filtering words starts')
    words = word_tokenize(text)
    words = wordsFilterIntext(words)
    print('filtering words ends')

    print('extracting feature starts')
    feature = buildClassifierFeature(words)
    print('extracting feature ends')

    features = []
    features.append(feature)
    print('text preprocessing ends')

    print('prediction starts')
    predictionHtml = ''
    for i in range(len(classifiers)):
        predictionHtml += classifierNames[i] + ' predict that text is ' + classifiers[i].predict(features)[0] + '<br>'
    print('prediction ends')

    return predictionHtml

def main():
##  to disable the warnings
    warnings.filterwarnings("ignore")

    if style == 0:
        '''
        dataset = loadData()
        random.shuffle(dataset)

        # # to get the part of dataset
        #dataset = dataset[:100]

        
        print('removing back discourse starts')
        global discourse
        global sentenceSpliter
        discourse = loadDiscourse()
        datasetLength = len(dataset)
        for l in range(datasetLength):
            isDiscourse = False
            length = len(dataset[l][0])
            newWords = []
            start = 0
            for i in range(length):
                if(isDiscourse and (dataset[l][0][i] in sentenceSpliter)):
                    start = i + 1
                    isDiscourse = False
                else:
                    for d in discourse:
                        if(dataset[l][0][i] == d[0]):
                            dLength = len(d)
                            isDiscourse = True
                            for j in range(1, dLength):
                                if((not dataset[l][0][i + j]) or dataset[l][0][i + j] != d[j]):
                                    isDiscourse = False
                                    break

                            if(isDiscourse):
                                #print('match discourse :', d, ' ', dataset[l][0][i], 'in line', l)
                                newWords.extend(dataset[l][0][start:i])
                                break
            
            if(not isDiscourse):
                newWords.extend(dataset[l][0][start:])
            dataset[l][0] = newWords
            del newWords
        print('removing back discourse ends')
        

        # to filter data
        print('filtering data start')
        global vocabulary
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
        save(dataset, dataFileNames[isMovieReviews])
        print('saving filtered dataset end')

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
        
        print('test targeted classifier features start')
        testClassifierFeatures, testTargets = buildTargetedClassifierFeatures(testData)
        print('test targeted classifier features are completed')

        print('saving test classifier features and test targets start')
        save(testClassifierFeatures, 'TestClassifierFeatures')
        save(testTargets, 'TestTargets')
        print('saving test classifier features and test targets end')
        '''

        
        print('loading features and targets starts')
        trainingClassifierFeatures = load('TrainingClassifierFeatures')
        trainingTargets = load('TrainingTargets')

        testClassifierFeatures = load('TestClassifierFeatures')
        testTargets = load('TestTargets')

        # trainingClassifierFeatures = trainingClassifierFeatures[100:200]
        # testClassifierFeatures = testClassifierFeatures[10:20]

        # trainingTargets = trainingTargets[100:200]
        # testTargets = testTargets[10:20]

        # trainingDataSize = 100
        # testDataSize = 10

        trainingDataSize = len(trainingTargets)
        testDataSize = len(testTargets)
        print('loading features and targets ends')
        
        '''
        if(trainingMode != 0):
            trainingClassifierFeatures = trainingClassifierFeatures[trainingSize: trainingSize * 2]
            trainingTargets = trainingTargets[trainingSize: trainingSize * 2]
            trainingDataSize = trainingSize
        
        
        # to train the sentiment models
        global classifiers
        gc.collect()

        if(trainingMode == 0):
            # MultinomialNB
            classifiers.append(MultinomialNB())
            # BernoulliNB
            classifiers.append(BernoulliNB())
            # Stochastic Gradient Descent
            classifiers.append(SGDClassifier())
        elif(trainingMode == 1):
            # LogisticRegression abort because it doesn't have 'partial_fit'
            classifiers.append(LogisticRegression())
        elif(trainingMode == 2):
            # SVC no partial_fit
            classifiers.append(SVC())
            # LinearSVC no partial_fit
            classifiers.append(LinearSVC())
            # NuSVC no partial_fit
            classifiers.append(NuSVC())
        
        if(trainingMode == 0):
            # to partially train the classifier
            print('classifiers partial training start')
            start = 0
            end = 0
            tempSize = trainingDataSize
            # it is passed to partial_fit when first call
            allTargets = np.array(['neg', 'pos'])

            # first call
            if(tempSize > trainingSize):
                end += trainingSize
                for classifier in classifiers:
                    # print('i = ', i)
                    classifier.partial_fit(trainingClassifierFeatures[start: end], trainingTargets[start: end], classes = allTargets)

                tempSize -= trainingSize
                start = end

            # to partially train continually
            while(tempSize > trainingSize):
                end += trainingSize
                for classifier in classifiers:
                    classifier.partial_fit(trainingClassifierFeatures[start: end], trainingTargets[start: end])

                tempSize -= trainingSize
                start = end

            # last call
            end += tempSize
            for classifier in classifiers:
                classifier.partial_fit(trainingClassifierFeatures[start: end], trainingTargets[start: end], classes = allTargets)

            # to use our vote classifier
            # can use function like 'add' to add classifier?
            voteClassifier = VoteClassifier(classifiers[0],
                                            classifiers[1],
                                            classifiers[2])
            classifiers.append(voteClassifier)
            
            print('classifiers partial training ends')
        elif(trainingMode == 1):
            print('classifiers fitting starts')
            # for classifier in classifiers:
            for i in range(len(classifiers)):
                print('i = ', i)
                classifiers[i].fit(trainingClassifierFeatures, trainingTargets)
                save(classifiers[i], classifierNames[i])

            voteClassifier = VoteClassifier(classifiers[0])
            classifiers.append(voteClassifier)
            print('classifiers fitting ends')
        elif(trainingMode == 2):
            print('classifiers fitting starts')
            # for classifier in classifiers:
            for i in range(len(classifiers)):
                print('i = ', i)
                classifiers[i].fit(trainingClassifierFeatures, trainingTargets)
                save(classifiers[i], classifierNames[i])

            voteClassifier = VoteClassifier(classifiers[0])
            classifiers.append(voteClassifier)
            print('classifiers fitting ends')
        
        print('saving classifiers starts')
        if(trainingMode == 0):
            temp = len(classifiers) - 1
            for i in range(temp):
                save(classifiers[i], classifierNames[i])
        elif(trainingMode == 1):
            print('saved')
            # temp = len(classifiers) - 1
            # for i in range(temp):
            #     save(classifiers[i], classifierNames[i])
        elif(trainingMode == 2):
            print('saved')
        print('saving classifiers ends')
        '''


        print('loading classifiers starts')
        global classifiers

        if(trainingMode == 0):
            temp = len(classifierNames) - 1
            for i in range(temp):
                classifiers.append(load(classifierNames[i]))

            voteClassifier = VoteClassifier(classifiers[0],
                                            classifiers[1],
                                            classifiers[2])
            classifiers.append(voteClassifier)
        elif(trainingMode == 1):
            temp = len(classifierNames) - 1
            for i in range(temp):
                classifiers.append(load(classifierNames[i]))

            voteClassifier = VoteClassifier(classifiers[0],
                                            classifiers[1],
                                            classifiers[2])
            classifiers.append(voteClassifier)
        elif(trainingMode == 3):
            temp = len(classifierNames) - 1
            for i in range(temp):
                classifiers.append(load(classifierNames[i]))

            voteClassifier = VoteClassifier(classifiers[0],
                                            classifiers[1],
                                            classifiers[2],
                                            classifiers[3],
                                            classifiers[4],
                                            classifiers[5],
                                            classifiers[6])
            classifiers.append(voteClassifier)

        print('loading classifiers ends')
        

        # to predict
        print('predicting start')
        gc.collect()

        for i in range(len(classifiers)):
            print(classifierNames[i], 'Algo accuracy percent: ', (accuracy(classifiers[i], testClassifierFeatures, testTargets)) * 100)
            # print("Classification: ", voteClassifier.classify(test[0]), " Confidence: ", voteClassifier.confidence(test[0]), 'in real: ', test[1])

        # to save the classifier as pickle except vote classifier
        # temp = classifierNumber - 1
        #for i in range(classifierNumber):
        #    save(classifiers[i], classifierNames[i])
        
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
        print('Module mode')

        # print('loding training classifier features and training targets start')
        # trainingClassifierFeatures = load('TrainingClassifierFeatures')
        # trainingTargets = load('TrainingTargets')
        # print('loding training classifier features and training targets end')

        # print('loading test classifier features and test targets start')
        # testClassifierFeatures = load('TestClassifierFeatures')
        # testTargets = load('TestTargets')
        # print('loading test classifier features and test targets end')

        print('loading vocabulary starts')
        # global vocabulary
        vocabulary = load('Vocabulary')
        print('loading vocabulary ends')

        print('loading classifiers starts')
        temp = len(classifierNames) - 1
        for i in range(temp):
            classifiers.append(load(classifierNames[i]))

        voteClassifier = VoteClassifier(classifiers[0],
                                        classifiers[1],
                                        classifiers[2])
        classifiers.append(voteClassifier)
        print('loading classifiers ends')
        
        print('twitter tren strarts')

        reviews, hashtable = twittertrend.trend(number = 5, lag = 0)

        # reviews is [[]]
        for review in reviews:
          print(review)

        print('twitter trend ends')

        print('twitter parsetw starts')
        
        result = stream.parsetw(track = 'Trump', number = 2)
        
        # result is [[[]]]
        for r in result[0]:
          print(r)

        print('twitter parsetw ends')

        print('twitter search starts')
        
        texts, tweets = search.twittersearch(q = ['War'], count = 10)
        
        # texts is []
        for text in texts:
          print('text: ' + text)
        
        print('twitter search ends')

        # print('predicting start')

        # for i in range(len(classifiers)):
        #     print(classifierNames[i], 'Algo accuracy percent: ', (accuracy(classifiers[i], testClassifierFeatures, testTargets)) * 100)
        
main()