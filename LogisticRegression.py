import numpy as np
from numpy import log, dot, e

class LogisticRegression:
    def __init__(self, lr=0.05, epochs=100, intercept=True):
        self.lr = lr
        self.epochs = epochs
        self.intercept = intercept
        self.bias = 0
    
    def addIntercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.intercept:
            X = self.addIntercept(X)
        
        self.loss = []
        
        # weights initialization
        self.weight = np.zeros(X.shape[1])
        
        for i in range(self.epochs):
            z = np.dot(X, self.weight)
            h = self.sigmoid(z + self.bias)
            
            gradient = np.dot(X.T, (h - y)) / y.size
            bGradient = np.sum(h - y) / y.size
            
            self.weight -= self.lr * gradient
            self.bias -= self.lr * bGradient
            
            self.loss.append(self.cost(h, y))
    
    def predict_prob(self, X):
        if self.intercept:
            X = self.addIntercept(X)
    
        return self.sigmoid(np.dot(X, self.weight) + self.bias)
    
    def predict(self, X):
        return self.predict_prob(X) >= 0.5