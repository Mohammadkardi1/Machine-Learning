import random
import csv
import numpy as np


def loadCsv(filename):
    """Load the CSV file"""
    lines = csv.reader(open(filename))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]  # Convert String to Float numbers
    return dataset


def splitDataset(dataset,splitRatio):
    trainSize=int(len(dataset)*splitRatio)
    trainSet=[]
    copy=list(dataset)
    while len(trainSet) < trainSize:
        index=random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]


def separateByClass(dataset):
    """Split training set by class value"""
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def summarize(dataset):
    """Find the mean and standard deviation of each feature in dataset"""
    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
    del summaries[-1] #Remove last entry because it is class value.
    return summaries


def summarizeByClass(dataset):
    """find the mean and standard deviation of each feature in dataset by their class"""
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    """Calculate probability using gaussian density function"""
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent


def calculateClassProbabilities(summarizes, inputVector):
    """Calculate the class probability for input sample. Combine probability of each feature"""
    probabilities = {}
    for classValue, classSummarizes in summarizes.items():
        probabilities[classValue] = 1
        for i in range(len(classSummarizes)):
            mean, stdev = classSummarizes[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    """Compare probability for each class. Return the class label which has max probability."""
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None,-1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    """Get class label for each value in test set."""
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet,predictions):
    """Create a naive bayes model. Then test the model and returns the testing result."""
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct +=1
    return (correct/float(len(testSet)))*100.0


def main():
    # load and prepare data
    filename = '.\\banknote.csv'
    splitRatio=0.67
    dataset=loadCsv(filename)
    trainingSet,testSet=splitDataset(dataset,splitRatio)
    print('split {0} rows into train = {1} and test = {2} rows '.format(len(dataset),len(trainingSet),len(testSet)))
    #prepare model
    summaries = summarizeByClass(trainingSet)
    #test model
    predictions=getPredictions(summaries,testSet)
    accuracy=getAccuracy(testSet,predictions)
    print('Accuracy:  {0}%'.format(accuracy))
    
  
main()