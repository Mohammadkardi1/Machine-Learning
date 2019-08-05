#importing librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data
path = '.\\data.txt'
data = pd.read_csv(path,names=['Population','Profit'])

#draw data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(5,5))

# adding a new column called ones before the data
data.insert(0, 'Ones', 1)

# separate X (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

## convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)

#Calculates the cost for given X and Y
def  cal_cost(theta,X,y):

    m = len(y)
    
    predictions = X.dot(theta)
    cost = (1/2 * m) * np.sum(np.square(predictions-y))
    return cost

#SGD function
def stocashtic_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10):

    m = len(y)
    cost_history = np.zeros(iterations)
    
    
    for it in range(iterations):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost
        
    return theta, cost_history


theta = np.random.randn(2,1)
eta=0.01
max_iter = 1000

g, cost = stocashtic_gradient_descent(X, y, theta, eta, max_iter)
print('theta  =',g)

# get best fit line
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[1, 0] * x)

# draw the line
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(max_iter), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')