import numpy as np

from keras import backend as k

def create_adv_loss(discriminator):
    def loss(y_true, y_pred):
        return discriminator(y_pred)
    return loss

class model():
    def __init__(self, model = None,
                 optimizer = None,
                 loss = 'mean_squared_error',
                 verbose = 0,
                 discriminatormodel = None,
                 n = 0):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.n = n
        if discriminatormodel==None:
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
        else:
            self.generatorlossmodel = discriminatormodel
            self.loss = create_adv_loss(discriminatormodel)
            self.model.compile(loss=loss, optimizer=self.optimizer)#todo rimetti la loss come self.loss
        self.verbose = verbose
        #self.weights?
    def get_compiled_model(self):
        return self.model

    def evaluate(self, instance):
        predictions = self.model.predict([[instance]])
        return predictions[0, 0]
        # Compute Q(s,a) and return the its value

    def evaluaten(self, instance):
        predictions = self.model.predict([[instance]])
        return predictions[0]
        # Compute Q(s,a) and return the its value

    def update(self, minibatch = None):
        self.longUpdate(minibatch, len(minibatch), 1)
        return


    def longUpdate(self, minibatch, length, epochs):
        #print(minibatch,"before")
        minibatch = np.array(minibatch)
        #print(minibatch,"<==== after")
        n = len(minibatch[0, :])
        #print("nnn",n)
        X = minibatch[:, range(n - 1)]
        #print("X",X)
        Y = minibatch[:, n - 1]

        # print("_MINIBATCH ", n)
        # print("_MINIBATCH ", minibatch[0,range(n-1)])
        #print("_________X ", X)
        #print("_________Y ", Y)

        self.model.fit(X, Y, epochs=epochs, batch_size=length, verbose=self.verbose)
        return
    # Make one step

    def input_gradients(self, xk, x0):
        sess = k.get_session()
        input_gradients = k.gradients(self.model.output, self.model.input)
        input_grads = sess.run(input_gradients, feed_dict={self.model.input: np.array([np.concatenate([xk, x0])])})
        input_gradients = input_grads[0][0][len(xk):len(xk) * 2]
        return input_gradients

    def input_gradients_using_gradients(self, xk, x0, gradientk, gradient0, fk, f0):
        sess = k.get_session()
        input_gradients = k.gradients(self.model.output, self.model.input)
        input_grads = sess.run(input_gradients, feed_dict={self.model.input: np.array([np.concatenate([xk, x0, gradientk, gradient0, [fk], [f0]])])})
        input_gradients = input_grads[0][0][len(xk):len(xk) * 2]
        return input_gradients