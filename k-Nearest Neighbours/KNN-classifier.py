# Importing libraries
import pandas as pd
import numpy as np
import operator


# Importing data 
data = pd.read_csv(".//iris.txt")

# Importing data 
data = pd.read_csv(".//iris.txt")
data = np.array(data.values)


# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(x1,x2,n): # n is is Number of features
    distance = 0
    for i in range(n):
        distance += np.square(x1[i] - x2[i])
    return np.sqrt(distance)

# Defining our KNN model
def getNeighbors(trainingSet, x_test, k):
 
    distances = []
    length = len(x_test)
    
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        dist = euclideanDistance(x_test, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
        
        
    distances.sort(key=operator.itemgetter(1))
    neighbors = []  #neighbours index
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
    

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1] #classes names
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
 
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]




# Creating a dummy testset
testSet = [7.2, 3.6, 5.1, 2.5]

# Initialise the value of k
k = 3  

# Running KNN model
neighbors = getNeighbors(data, testSet, k) 

#3 nearest neighbors
neighbors=np.array(neighbors)
print(neighbors)

# Predicted class
result=getResponse(neighbors)
print(result)