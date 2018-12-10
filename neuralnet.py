"""
-----------------------------------------------------------------------
-------- SIMPLE NEURAL NETWORK MODEL FOR THE IRIS DATASET ------------
-----------------------------------------------------------------------
Created on Tue Aug 28 13:26:05 2018
@file: neuralnet.py
@language: Python 3.6.6
@author: Anthony Leung (leungant@yorku.ca)
"""
#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



##---------------------------------------------------------------##
##----------------------- DATA PROCESSING -----------------------##
##---------------------------------------------------------------##

from sklearn.datasets import load_iris
dataset = load_iris() #Load the iris data set
nSamples, nFeatures = dataset.data.shape #Extract number of samples and features

X = dataset.data        # The input values
y = dataset.target      # The output labels (in integers)

y = np.reshape(y,(nSamples,1))

# Split the data into training and testing sets
(Xtrain, Xtest, ytrain, ytest) = train_test_split(X, y, test_size=0.4)



##---------------------------------------------------------------##
##--------------------- AUXILLARY FUNCTIONS ---------------------##
##---------------------------------------------------------------##

def oneHotEncode(y):
    ''' 
    Routine to convert a label vector into one-hot format
    Input:
        y : Label vector
    Return:
        yOH : one-hot encode matrix
    '''
    nLabel = len(np.unique(y)) #Gets the number of unique labels in y
    y = np.asarray(y, dtype='int32')
    if len(y) > 1:
        y = y.reshape(-1)
    if not nLabel:
        nLabel = np.max(y) + 1
    yOH = np.zeros((len(y), nLabel))
    yOH[np.arange(len(y)), y] = 1
    return yOH


def softmax(z):
    '''
    Computes the softmax value
    Input:
        z : Input value/vector (z=wx)
    Output:
        Softmax probability
    '''
    return np.exp(z)/np.sum(np.exp(z),axis=1, keepdims=True)


def softmaxGrad(z, ytrain):
    '''
    Computes the softmax gradient value
    Input:
        z : Input value/vector (z=wx)
        ytrain : Training y-values
    Output:
        Softmax gradient
    '''
    p = softmax(z)
    grad = p - ytrain
    return grad


def sigmoid(z):
    '''
    Computes the sigmoid value
    Input:
        z : Input value/vector (z=wx)
    Output:
        Sigmoid probability
    '''
    return 1.0/(1.0+(np.exp(-z)))

    
def sigmoidGrad(z):
    '''
    Computes the sigmoid gradient value
    Input:
        z : Input value/vector (z=wx)
    Output:
        Sigmoid gradient
    '''
    f = sigmoid(z)
    return (f * (1.0 - f))



##----------------------------------------------------------------##
##--------------------- NEURAL NETWORK CLASS ---------------------##
##----------------------------------------------------------------##

class NeuralNet():
    '''
    Neural network (NN) class based on a softmax classifier. 
    
    Minimization of the cross-entropy loss of this 
    network involves the use of momentum.
    
    Activation of hidden layers uses a sigmoid function.
    '''
    def __init__(self, dims, momentum, lmda, epochs):
        self.nLayers       = len(dims)
        self.nInput        = dims[0]
        self.nHiddenLayers = len(dims) - 2
        self.nOutput       = dims[-1]
        
        self.momentum      = momentum
        self.lmda          = lmda # L2 weight decay regularization
        self.epochs        = epochs
        self.loss          = []   # To cache the train loss per epoch
        self.accuracy      = []   # To cache the test accuracy per epoch
        
        # Initialize weights with random values.
        # A multiplicative factor (Xavier heuristic: np.sqrt(1.0/y)) 
        # is applied to alleviate the problem of vanishing/exploding gradients.
        self.weights       = [np.random.randn(y, x) * np.sqrt(1.0/y) \
                                  for x, y in zip(dims[:-1], dims[1:])]
        
        # Initialize biases with random values.
        self.biases        = [np.random.randn(y, 1) for y in dims[1:]]

        # Initialize velocity with zeros
        self.velocity      = [np.zeros(w.shape) for w in self.weights]
        
        
    def crossEntropyLoss(self, y_out, ytrain):
        ''' Routine to compute the cross-entropy loss and its
            corresponding gradient wrt to the weight w and bias b.
        Input 
            y_out: Output values from forward propagation
            ytrain: y-values of the training data
        Output
            dw: gradient wrt w
            db: gradient wrt b
        '''
        m = len(ytrain)
        lbda = self.lmda  
        
        # Compute the total loss.
        regLoss = 0.5 * lbda * np.sum(self.weights[-1]*self.weights[-1])
        loss = -np.log(np.max(y_out)) * ytrain
        totalLoss = np.sum(loss)/m + regLoss
        self.loss.append(totalLoss) # Keep track of the loss per epoch
        
        # Compute the gradient wrt w (dw).
        dw = ((-1.0/m) * np.dot(self.activations[-2].T, (ytrain - y_out))) \
                                + lbda * self.weights[-1].T
                                
        # Compute the gradient wrt b (db).
        db = (-1.0/m) * np.sum(ytrain - y_out, axis=0)
        db = np.reshape(db,(len(db),1))
        
        return dw, db
      

    def forwardprop(self, x):
        ''' Forward propagation routine 
        Input:
            x: x-values from the data set (Can be test or train)
        Output:
            out: the output values of the NN
        '''
        atemp = x
        self.activations = [] #To cache activation results
        self.zs=[] # To cache z values where z = w.dot(x)
        
        # Input Layer
        self.activations.append(atemp)
        
        # Hidden Layer(s)
        iDepth = 1
        for w, b in zip(self.weights, self.biases):
            if iDepth > self.nHiddenLayers: 
            #If next layer is the output, break.
                break
            z = np.dot(atemp, w.T) + b.T
            self.zs.append(z)
            sigma = sigmoid(z)
            atemp = sigma
            self.activations.append(sigma)
            iDepth += 1
        
        # Output Layer
        z = np.dot(atemp, self.weights[-1].T) + self.biases[-1].T
        self.zs.append(z)
        sigma = softmax(z)
        self.activations.append(sigma)
            
        out = self.activations[-1] #Extract the output activations
        return out
    
    
    def backwardprop(self, y_out, ytrain):
        ''' Back propagation routine for gradient descent 
        Input:
            y_out: the output values from forward propagation
            ytrain: the y-values of the training set
        Output:
            nabla_w: the gradient of the loss wrt w
        '''
        # Initialize the gradient wrt to weights w and bias b.
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Start with the output layer to compute the loss and gradient wrt w.
        dw, db = self.crossEntropyLoss(y_out, ytrain)

        sp = softmaxGrad(self.zs[-1], ytrain)
        delta =  sp * self.activations[-1]
        nabla_w[-1] = dw.T
        nabla_b[-1] = db
        
        # Back propagate through the hidden layers
        for l in range(2,self.nLayers):
            z = self.zs[-l]
            sp = sigmoidGrad(z)
            delta = np.dot(delta,self.weights[-l+1]) * sp
            
            nabla_w[-l] = np.dot(self.activations[-l-1].T,delta).T
            db = (np.sum(delta, axis=0))
            db = np.reshape(db,(len(db),1))
            nabla_b[-l] = db
            
        return nabla_w, nabla_b
    
    
    def train(self, eta, xtrain, ytrain, xtest=None, ytest=None):
        ''' The main training routine to model the data 
        Input:
            eta : the learning rate
            xtrain, ytrain: training data 
            xtest, ytest: testing data
        Output:
            None
        '''
        # Training iteration
        for i in range(self.epochs):
            # Forward propagation
            y_out = self.forwardprop(xtrain)
            
            # Backward propagation
            nabla_w, nabla_b = self.backwardprop(y_out, ytrain)

            # Update the velocities, weights, and biases
            m = self.momentum
            self.velocity = [m * v + eta * dw for (v,dw) in zip(self.velocity, nabla_w)]
            self.weights = [w - v for (w, v) in zip(self.weights, self.velocity)]
            self.biases = [(b - eta * db) for (b, db) in zip(self.biases, nabla_b)]
            
            # Evaluate results with updated weights.
            if ytest.any() and xtest.any(): 
                score = self.evaluate(xtest, ytest)
                ratio = score/len(ytest)
                self.accuracy.append(ratio)
                
                if i == (self.epochs - 1): # At final epoch
                    print("\nRESULTS AT FINAL EPOCH\n")
                    print("Training loss: {0:.6f}".format(self.loss[-1]))
                    print("Prediction score: {}/{}".format(int(score),len(ytest)))

    
    def evaluate(self, xtest, ytest):
        ''' Routine to evaluate the NN model by using test data
        Input: 
            xtest: x-values from the test set
            ytest: y-values from the test set
        Output:
            score: the total number of correct predictions
        '''
        
        # Forward propagate with xtest
        yout = self.forwardprop(xtest)
        
        # Process probability results into one-hot format by
        # finding the position of max value and assign to 1 in ypredict.
        ypredict = np.zeros_like(yout)

        ypredict[np.arange(len(yout)), yout.argmax(1)] = 1

        score = 0
        # Evaluate predictions of the NN model
        for u, v in zip(ypredict, ytest):
            score += np.dot(u,v)
        
        return score


    def plotLossGraph(self):
        '''Simple routine to plot the loss over time'''
        loss = self.loss
        plt.plot(loss)
        plt.title("Time evolution of cross-entropy (training) loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.show()
        
        
    def plotAccuracyGraph(self):
        '''Simple routine to plot the test accuracy over time'''
        acc = self.accuracy
        plt.plot(acc)
        plt.title("Time evolution of test accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Score ratio")
        plt.show()
        
        

##------------------------------------------------------------##
##--------------------- MAIN APPLICATION ---------------------##
##------------------------------------------------------------##
if __name__ == '__main__':
    #---------------------------------------
    # Parse user arguments, if applicable.
	#---------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-hl","--hiddenlayer",dest="hiddenLayer", type=int, nargs='+',
                    help='Array of node numbers of hidden layers')
    parser.add_argument("-lr", "--learningrate", dest="learningRate", default=0.05,
                        type=float, help="Learning rate of the neural network")
    parser.add_argument("-m", "--momentum", dest="momentum", default=0.06,
                        type=float, help="Momentum parameter")
    parser.add_argument("-rg", "--regularization", dest="regularization", default=0.005,
                        type=float, help="Momentum parameter")    
    parser.add_argument("-ep", "--epoch", dest="epochs", default=5000,
                        type=int, help="Total epochs")

    args = parser.parse_args()
    
    eta = args.learningRate
    momentum = args.momentum
    lmda = args.regularization
    epochs = args.epochs
    
    print("\nPARAMETERS FOR NEURAL NETWORK TRAINING\n")
    print("Learning rate (eta): ", eta)
    print("Momentum: ", momentum)
    print("L2 regularization (lambda): ", lmda)
    print("Total epochs assigned: ", epochs)
    
    # Populate the dimension array based on number of nodes per layer.
    # Configuration:
    #   dims = [#NodeInputs, #NodesHL1, #NodesHL2,...,#NodeOutputs]
    dims=[] 
    
    dims.append(nFeatures) #Number of input classes appended
    if (args.hiddenLayer): #Hidden layers, if applicable
        print("Hidden layer configuration: ", args.hiddenLayer)
        for x in args.hiddenLayer:
            dims.append(int(x)) #Number of Nodes per Hidden Layer appended
    
    # Convert y-label into one-hot format.
    ytrain = oneHotEncode(ytrain)
    ytest = oneHotEncode(ytest)
    
    dims.append(len(ytrain[0])) #Number of labels for the output appended

    #---------------------------------------------
    # Create and train neural network.
    #---------------------------------------------
    NN = NeuralNet(dims, momentum, lmda, epochs)
    NN.train(eta, Xtrain, ytrain, Xtest, ytest)
    # Plot results.
    NN.plotLossGraph()
    NN.plotAccuracyGraph()
 
    
