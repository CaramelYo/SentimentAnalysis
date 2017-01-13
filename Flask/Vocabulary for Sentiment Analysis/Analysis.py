import nltk

from nltk.corpus import stopwords
from nltk.data import load
from nltk.tokenize import word_tokenize

##global info
vocabulary = []
stopWords = stopwords.words('english')

def main():
##    with open('NegativeEvaluation.txt', 'r') as inputFile:
####      to skip first two rows
##        inputFile.readline()
##        inputFile.readline()
##
##        for line in inputFile:
####            print(line)
##            words = word_tokenize(line)
####            print(words)
##
##            for word in words:
##                if word not in stopWords and word not in vocabulary:
##                    vocabulary.append(word)
##
####            print(words)
####            
####            break
##
##
####      to output to txt
##        with open('Negative.txt', 'w') as outputFile:
##            for word in vocabulary:
##                print(word, file = outputFile)

    tagdict = load('help/tagsets/upenn_tagset.pickle')
##    for tag in tagdict.keys():
##        print(tag, '       ', tagdict[tag][0])
    print(tagdict['PRP$'][1])

main()
