import numpy as np
import math
from memory import memory
from rl import rl
import keras as ks
from keras.models import Sequential
from keras.layers import Dense
from model import model

from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects




def createModel( layers, instance_dimension ):
    model = Sequential()


    tup = layers[0]
    model.add(Dense(tup[0], input_dim=instance_dimension, activation=tup[2],  kernel_initializer=tup[1]))
    for i in range(1, len(layers)):
        tup = layers[i]
        model.add(Dense( tup[0], activation=tup[2],  kernel_initializer=tup[1]))
    #optimizer =  ks.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.1, nesterov=True)
    optimizer = ks.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    return model, optimizer


def createGeneratorModel( layers, instance_dimension ):
    model = Sequential()

    tup = layers[0]
    model.add(Dense(tup[0], input_dim=instance_dimension, activation=tup[2],  kernel_initializer=tup[1]))
    for i in range(1, len(layers)):
        tup = layers[i]
        model.add(Dense( tup[0], activation=tup[2],  kernel_initializer=tup[1]))
    #optimizer =  ks.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.1, nesterov=True)
    optimizer = ks.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    return model, optimizer



def gradientMethod(x0, A, c, maxiter = 100000, learningrate = 0.0001):
    xk = np.array(x0)
    fk = rl.f(xk, A, c)
    #todo transform matrix A into np array or matrix
    xstar = xk
    fstar = fk
    stopcondition = "notinitialized"
    iter = -1
    xk1 = xk.copy()
    fk1 = fk
    while True and iter <= maxiter:
        iter += 1
        gradient = rl.nabla(xk,A, c)
        rate = learningrate/(iter+1.1)**0.2
        #print(rate)

        xk1 = xk - rate*gradient
        fk1 = rl.f(xk1, A, c)
        stopcondition = rl.stop( gradient, xk, xk1, fk, fk1 )
        if stopcondition != "notsatisfied":
            xstar = xk1
            fstar = fk1
            break
        xk = xk1
        fk = fk1
    if stopcondition == "notsatisfied" and iter >= maxiter:
        stopcondition = "maximumIterations "+str(iter)
        xstar = xk1
        fstar = fk1


    return fstar, xstar, iter, stopcondition

def custom_activation(x):
    return K.exp(-(x*x))




def main():
    get_custom_objects().update({'quadG': Activation(custom_activation)})


    n = 10
    np.random.seed( 310594 )

    x0 = np.random.rand( n )
    A = np.random.rand( n, 2*n)
    A = np.dot( A, np.transpose(A))
    c = np.random.randint( low = 1, high = 10, size=n)
    fstar, xstar, iter, stopcondition = gradientMethod(x0, A, c)
    print( "GRADIENT METHOD\nXstar", xstar,"\n Fstar", fstar,"\n Iter", iter,"\n StopCondition ", stopcondition)

    xn =-np.dot( np.linalg.inv(A), c)/2
    fn = rl.f(xn, A, c)
    print("NEWTON METHOD\nXstar", xn, "\n Fstar", fn)

    layers = [
        (40, 'uniform', 'relu'),
        (1800, 'uniform', 'relu'),
        # (200, 'uniform', 'relu'),
        (40, 'uniform', 'relu'),
        (1, 'uniform', 'linear')
    ]

    lays = [
        (40, 'uniform', 'relu'),
        (1800, 'uniform', 'relu'),
        # (200, 'uniform', 'relu'),
        (40, 'uniform', 'relu'),
        (n, 'uniform', 'linear')
    ]


    basic, optimizer = createModel(layers, instance_dimension=2*n)
    clmod = model(basic, optimizer, n=n)
    genbasic, genoptimizer = createModel(lays, instance_dimension= n)
    genmod = model( genbasic, genoptimizer, discriminatormodel=clmod.get_compiled_model(), n=n)

    smartgrad = rl(
                 clmod,
                 n,
                 episodes = 100,
                 steps = 50,
                 initial_probability = 0.8,
                 probability_decrease = 0.9,
                 future_relevance = 0.9,
                 experience = 0,
                 batch_size = 10,
                 npseed = None,
                 verbose = False,
                 starting_optimization_rate = 1,
                 trust_region=1e-1)
    smartgrad.feedGenerator(generatormodel=genmod)
    #fstar, xstar, iter, stopcondition, episode = smartgrad.solve(x0, A, c)
    fstar, xstar, iter, stopcondition, episode = smartgrad.solve_by_generator(x0, A, c)
    print("SMART GRADIENT WITH GRADIENT METHOD\nXstar", xstar,
          "\n Fstar", fstar, "\n Iter", iter, "\n StopCondition ", stopcondition)

    # basic, optimizer = createModel(layers, instance_dimension=2 * n )
    # clmod = model(basic, optimizer)
    # smartgrad = rl(
    #     clmod,
    #     n,
    #     episodes=5,
    #     steps=50,
    #     initial_probability=0.5,
    #     probability_decrease=0.5,
    #     future_relevance=0.9,
    #     experience=2,
    #     batch_size=4,
    #     npseed=None,
    #     verbose=False,
    #     starting_optimization_rate=1e-1,
    #     trust_region=1e-1)
    # fstar, xstar, iter, stopcondition, episode = smartgrad.solve(x0, A, c)
    # print("SMART GRADIENT METHOD\nXstar", xstar, "\n Fstar", fstar, "\n Iter", iter, "\n StopCondition ", stopcondition)
    #

main()





