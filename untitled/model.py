import numpy as np


class model():
    def __init__(self, model = None,
                 optimizer = None,
                 loss = 'mean_squared_error',
                 verbose = 0):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(loss = self.loss, optimizer = self.optimizer)
        self.verbose = verbose
        #self.weights?

    def evaluate(self, instance):
        predictions = self.model.predict([[instance]])
        #print("<>",predictions[0,0])
        return predictions[0,0]
        #Compute Q(s,a) and return the its value

    def update(self, minibatch = None):
        self.longUpdate(minibatch, len(minibatch), 1)
        return

    def longUpdate(self, minibatch, length, epochs):
        minibatch = np.array(minibatch)
        n = len(minibatch[0, :])
        X = minibatch[:, range(n - 1)]
        Y = minibatch[:, n - 1]

        # print("_MINIBATCH ", n)
        # print("_MINIBATCH ", minibatch[0,range(n-1)])
        # print("_________X ", X)
        # print("_________Y ", Y)

        self.model.fit(X, Y, epochs=epochs, batch_size=length, verbose=self.verbose)
        return
    # Make one step