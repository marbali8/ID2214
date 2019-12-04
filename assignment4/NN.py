import numpy as np

# from https://enlight.nyc/projects/neural-network/
class OneLayerNN(object):

    def __init__(self, inputsize):

        #parameters
        self.inputSize = inputsize
        self.outputSize = 1
        self.hiddenSize = 20

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # from hidden to output layer

    def forward(self, X):

        #forward propagation through our network
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z) # activation
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation
        return o

    def sigmoid(self, s): # activation function

        return 1/(1 + np.exp(-s))

    def sigmoidPrime(self, s): # derivative of sigmoid

        return s * (1 - s)

    def backward(self, X, y, o): # backward propagate through the network

        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

    def train(self, X, y):

        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self, toPredict):

        # print("Predicted data based on trained weights: ")
        # print("Input (scaled): \n" + str(toPredict))
        print("Output unique: \n" + str(np.unique(self.forward(toPredict))))
