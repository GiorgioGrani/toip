import numpy as np
from keras import backend as k
import tensorflow as tf

class balerionmodel():
    def __init__(self, model = None,
                 optimizer = None,
                 loss = 'mean_squared_error',
                 verbose = 0,
                 learningrate = 0.001):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(loss = self.loss, optimizer = self.optimizer)
        self.verbose = verbose
        self.learningrate = learningrate

        #self.weights?

    def get_model(self):
        return self.model

    def evaluate(self, instance):
        predictions = self.model.predict([[instance]])
        #print("                                                           <>",predictions)
        return predictions[0,0]
        #Compute Q(s,a) and return the its value

    def multievaluate(self, instance):
        #print(instance,"INSTANCEEEEEEE")
        predictions = self.model.predict([[instance]])
        #print("<>",predictions[0,0])
        return predictions[0]
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

    def innerouterupdate(self,minibatch, inner_model, action_size_n, state_size, kk):
        evaluated_gradients = []
        sess = k.get_session()
       # # print(minibatch,"MINIBATCH")
       #  for instanceR in minibatch:
       #      onehotria = instanceR[0:action_size_n].copy()
       #      state = instanceR[action_size_n:state_size+action_size_n].copy()
       #      act = instanceR[state_size + action_size_n: state_size+ 2*action_size_n].copy()
       #      nstate = instanceR[state_size+ 2*action_size_n: 2*state_size+ 2*action_size_n].copy()
       #      instance = np.concatenate([state, act, nstate])
       #      rew = instanceR[len(instanceR) - 1].copy()
       #
       #      #act = inner_model.multievaluate(state)
       #
       #      print("State", state,"\nnstate", nstate,"action", act,"\nrew",rew,"\ninsranceR", instanceR,"\n\n")
       #
       #
       #
       #      #print(instanceR, "INSTANCE")
       #      #print(instance, "INSTANCE")
       #
       #
       #
       #
       #      eval_function = self.evaluate(instance)
       #      listOfVariableTensors = self.model.trainable_weights
       #      gradients = k.gradients(self.model.output, listOfVariableTensors)
       #      if evaluated_gradients == []:
       #          d_gradients = sess.run(gradients, feed_dict={self.model.input: np.array([instance])})
       #          for grad in d_gradients:
       #              evaluated_gradients.append( grad*2*(rew - eval_function))
       #
       #      else:
       #          cop = evaluated_gradients.copy()
       #          eval = sess.run(gradients, feed_dict={self.model.input: np.array([instance])})
       #          evaluated_gradients = []
       #          for grad in zip( cop, eval):
       #              evaluated_gradients.append( grad[0] + 2*(rew - eval_function)*grad[1] )
       #  inner_gradients = evaluated_gradients
       #  newweights = []
       #  w = self.model.get_weights()
       #  for val in zip( w, inner_gradients):
       #      newweights.append(val[0] + (val[1])*self.learningrate/kk)


        self.update(minibatch)

        expected = 0
        lll = len(minibatch)


        inner_final_gradients = []
        innermodel = inner_model.get_model()
        count = -1

        for instanceR in minibatch:
            count = count + 1
            #modify the inner model to get along with the new configuration


            onehotria = instanceR[0:action_size_n].copy()
            state = instanceR[action_size_n:state_size+action_size_n].copy()
            act = instanceR[state_size + action_size_n: state_size+ 2*action_size_n].copy()
            nstate = instanceR[state_size+ 2*action_size_n: 2*state_size+ 2*action_size_n].copy()
            instance = np.concatenate([onehotria,state, act, nstate])
            rew = instanceR[len(instanceR) - 1].copy()
            eval_func = self.evaluate(instance)

            expected = expected + self.evaluate(np.concatenate([onehotria,state, inner_model.multievaluate(np.concatenate([state, onehotria])), nstate]))
            print("move", np.around(inner_model.multievaluate(np.concatenate([state, onehotria]))))
            print("move", (inner_model.multievaluate(np.concatenate([state, onehotria]))))

            #print(",,minibatch instance",count,"value", instanceR)

            input_gradients = k.gradients(self.model.output, self.model.input)
            input_grads = sess.run(input_gradients, feed_dict={self.model.input: np.array([instance])})
            input_gradients = input_grads[0][0][state_size : (state_size + action_size_n)]

            #print(input_gradients,"INPUT_GRADIENTS\ninput_grads", input_grads[0][0])

            inner_gradients = []

            for i in range(action_size_n):
                gradients = k.gradients(innermodel.output[:, i], innermodel.trainable_weights)
                gradients = sess.run(gradients, feed_dict ={innermodel.input : np.array([np.concatenate([state, onehotria])])})

                if inner_gradients == []:
                    for grad in gradients:
                        inner_gradients.append( 2*(rew-eval_func)*input_gradients[i]*grad)
                else:
                    inn = inner_gradients.copy()
                    inner_gradients = []
                    for grad in zip(inn,gradients):
                        inner_gradients.append(grad[0] + 2*(rew - eval_func)*input_gradients[i]*grad[1])
            if inner_final_gradients == []:
                inner_final_gradients = inner_gradients
            else:
                inn = inner_final_gradients.copy()
                inner_final_gradients = []
                for grad in zip(inn, inner_gradients):
                    inner_final_gradients.append( grad[0] + grad[1])


        newthetas = []
        o = innermodel.get_weights().copy()
        for val in zip(o, inner_final_gradients):
            newthetas.append(val[0] + val[1] * self.learningrate/kk)

        #self.model.set_weights(newweights)
        innermodel.set_weights(newthetas)

        mm = -1e6
        for zz in innermodel.get_weights():
            maz = np.max(zz)
            if maz >= mm:
                mm = maz

        mmm = -1e6
        for zz in self.model.get_weights():
            maz = np.max(zz)
            if maz >= mmm:
                mmm = maz



        print("                                                     >>>>>>EXPECTED----", expected,    "    kk",kk)
        #print("                                                     INPUT_GRADIENTS-->", input_gradients,"kk",kk, "lr/kk", self.learningrate/kk)


        #print("                         NEWEIGHTS->",mmm)
        print("                         NEWTHTAS->", mm)
        #sess.close()
        return

