from memory import memory
#from model import model
from model import model
from time import time
import sys
import math
from docplex.mp.model import Model

import numpy as np
import keras


class rl():
    def __init__(self,
                 model,
                 n,
                 episodes=100,
                 steps=100,
                 initial_probability=0.8,
                 probability_decrease=0.5,
                 future_relevance=0.9,
                 experience=1,
                 batch_size=10,
                 npseed=None,
                 verbose=False,
                 starting_optimization_rate=1e-2,
                 trust_region=0.5,
                 watchdog=10):
        self.model = model
        self.n = n
        self.episodes = episodes
        self.step = steps
        self.initial_probability = initial_probability
        self.probability_decrease = probability_decrease
        self.experience = experience
        self.batch_size = batch_size
        self.npseed = npseed
        self.verbose = verbose
        self.memory = memory()
        self.future_relevance = future_relevance
        self.starting_optimization_rate = starting_optimization_rate
        self.trust_region = trust_region
        self.watchdog = watchdog
        self.alpha_starter = self.starting_optimization_rate

    def feedGenerator(self, generatormodel):
        self.generatormodel = generatormodel


    def feedAandC(self, A, c):
        self.A = A
        self.c = c

    def f(x, A, c):
        return np.dot(np.dot(np.transpose(x), A), x) + np.dot(np.transpose(c), x)

    def nabla(x, A, c):
        # print(2*np.dot(A, x))
        return 2 * np.dot(A, x) + c

    def stop(gradient, xk, xk1, fk, fk1, epsgrad=1e-5, epstep=1e-15, epsfunc=1e-5):
        normgrad = np.linalg.norm(gradient)
        normstep = np.linalg.norm(xk - xk1)
        normfunc = np.absolute(fk - fk1)
        if (normgrad <= epsgrad):
            return "gradient " + str(normgrad)
        if (normstep <= epstep):
            return "step " + str(normstep)
        if (normfunc <= epsfunc):
            return "function " + str(normfunc)
        return "notsatisfied"

    def updateseed(self, val):
        if self.npseed != None:
            self.npseed = self.npseed + val
            np.random.seed(self.npseed)

    def _select_action(self, xk, optimization_rate, probability ):
        if np.random.rand() <= probability:
            xk1 = xk +np.random.rand(len(xk))*self.trust_region/2 - self.trust_region
            return xk1
        gradients = self.model.input_gradients( xk , xk )#.copy()
        alpha = self.alpha_starter
        gamma = 0.9 #todo che roba è questa?? Armijo remove pls
        theta = 0.5 #todo ma pure questo che è????
        maxiter = 1000
        ind = 0
        xk1 = xk - optimization_rate*gradients
        fk = self.model.evaluate(np.concatenate([xk, xk]))
        norm = np.dot(gradients, gradients)
        #print("____________________________________________________________________norm: ", norm)
        while self.model.evaluate(np.concatenate([xk, xk1])) > (fk - alpha*gamma*norm) : #here we are minimizing the model
            alpha = theta*alpha
            xk1 = xk - alpha * gradients
            ind = ind + 1
            if ind > maxiter :
                print("ERRORE MAXITER ",ind," alpha ",alpha)
                xk1 = xk - optimization_rate * gradients
                break
        #print(alpha,"    ",(alpha == self.alpha_starter),"   ", self.alpha_starter,"   ", ind)
        if alpha == self.alpha_starter:
            self.alpha_starter = self.alpha_starter*2
        print("____________________________________________________________________norm: ", norm, "alpha:", alpha," normx:", np.linalg.norm((xk1-xk)))

        return xk1

    def _select_action_by_generator(self, xk, optimization_rate, probability ):
        if np.random.rand() <= probability:
            xk1 = xk +np.random.rand(len(xk))*self.trust_region/2 - self.trust_region
            return xk1
        xk1 = self.generatormodel.evaluaten(np.array(xk))
        print("XK1",xk1)
        return xk1

    def _select_action_with_gradients(self, xk, gradientk, fk,optimization_rate, probability ):
        if np.random.rand() <= probability:
            xk1 = xk +np.random.rand(len(xk))*self.trust_region/2 - self.trust_region
            return xk1
        gradients = self.model.input_gradients_using_gradients( xk , xk, gradientk, gradientk, fk, fk )#.copy()
        xk1 = xk - optimization_rate*gradients #here we are minimizing the model
        return xk1

    def solve(self, x0, A, c):
        self.feedAandC(A,c)

        optimization_rate = self.starting_optimization_rate
        xstar = []
        fstar = 0
        iter = 0
        stopcondition = "notinitialized"

        for episode in range(self.episodes):
            self.updateseed(1)

            xk = []
            # if episode == 0:
            xk = np.array(x0)  +np.random.rand(self.n)*self.trust_region/2 - self.trust_region
            # else:
            # xk = xk1 + np.random.rand(self.n)-0.5

            gradientk = rl.nabla(xk, A, c)
            xk1 = xk.copy()
            fk = rl.f(xk, A, c)
            fk1 = rl.f(xk1, A, c)
            r = fk

            for step in range(self.step):
                # optimization_rate = self.starting_optimization_rate/(iter+1)
                ind = episode  # temporary choice
                iter += 1
                xk1 = self._select_action(xk, optimization_rate,
                                          self.initial_probability * (self.probability_decrease) ** ind)
                fk1 = rl.f(xk1, A, c)
                gradientk1 = rl.nabla(xk1, A, c)

                rew = fk1 - fk  # - np.linalg.norm(gradientk1) #important if you have to maximize or minimize
                r += rew






                # prew = rew
                #stopcondition = rl.stop(rl.nabla(xk1, A, c), xk, xk1, fk, fk1)
                stopcondition = "notsatisfied"
                if stopcondition != "notsatisfied":
                    xstar = xk1
                    fstar = fk1
                    print("STOP CONDITION",stopcondition)
                    break
                xs = xk1.copy()  # support value
                fs = fk1  # support value
                #gradients = gradientk1.copy()
                for future in range(self.experience):
                    xs1 = self._select_action(xs, optimization_rate,
                                              0)#self.initial_probability * (self.probability_decrease) ** (ind))
                    # rew = self.model.evaluate(np.concatenate([state, action, next_state]))
                    # print("----->",  np.concatenate[xs, xs1])

                    fs1 = rl.f(xs1, A, c)
                    #gradients1 = rl.nabla(xs1, A, c)
                    #gs = self.model.evaluate(np.concatenate([xs, xs1]))
                    rew = rew + (self.future_relevance**(future+1)) * (fs1 - fs)
                    #rew = rew + (self.future_relevance**(future+1)) * gs
                    xs = xs1
                    fs = fs1
                    #gradients = gradients1
                # print("INSTANCE", np.concatenate([xk, xk1, [r]]))

                print("ep", episode, " step", step, "function", fk1, "rew", rew," percrew",rew/max(abs(fk), 1e-6), "r", r, "probability",
                      self.initial_probability * (self.probability_decrease) ** ind, "norm", np.linalg.norm(gradientk1))

                self.memory.add(np.concatenate([xk, xk1, [rew/max(abs(fk), 1e-6)]]))

                batch_size = self.batch_size
                if ((self.memory.depth) < self.batch_size):
                    batch_size = self.memory.depth
                self.model.update(self.memory.extract_minibatch(batch_size))
                xk = xk1
                fk = fk1
                # r = 0 #understand if this is valuable or not
        if stopcondition == "notsatisfied":
            stopcondition = "maximumIterations " + str(iter)
            xstar = xk1
            fstar = fk1
        elif stopcondition == "notinitialized":
            stopcondition = "ERROR " + str(iter)
            xstar = xk1
            fstar = fk1

        return fstar, xstar, iter, stopcondition, episode

    def solve_by_generator(self, x0, A, c):
        self.feedAandC(A,c)

        optimization_rate = self.starting_optimization_rate
        xstar = []
        fstar = 0
        iter = 0
        stopcondition = "notinitialized"

        for episode in range(self.episodes):
            self.updateseed(1)
            xk = np.array(x0)  +np.random.rand(self.n)*self.trust_region/2 - self.trust_region
            xk1 = xk.copy()
            fk = rl.f(xk, A, c)
            fk1 = rl.f(xk1, A, c)
            r = fk

            for step in range(self.step):
                ind = episode  # temporary choice
                iter += 1
                xk1 = self._select_action_by_generator(xk, optimization_rate,
                                          self.initial_probability * (self.probability_decrease) ** ind)
                fk1 = rl.f(xk1, A, c)
                gradientk1 = rl.nabla(xk1, A, c)

                rew = fk1 - fk  # - np.linalg.norm(gradientk1) #important if you have to maximize or minimize
                r += rew

                stopcondition = "notsatisfied"
                if stopcondition != "notsatisfied":
                    xstar = xk1
                    fstar = fk1
                    print("STOP CONDITION",stopcondition)
                    break
                xs = xk1.copy()  # support value
                fs = fk1  # support value
                for future in range(self.experience):
                    xs1 = self._select_action_by_generator(xs, optimization_rate,
                                              self.initial_probability * (self.probability_decrease) ** (ind))
                    fs1 = rl.f(xs1, A, c)
                    rew = rew + (self.future_relevance**(future+1)) * (fs1 - fs)
                    xs = xs1
                    fs = fs1

                print("ep", episode, " step", step, "function", fk1, "rew", rew," percrew",rew/max(abs(fk), 1e-6), "r", r, "probability",
                      self.initial_probability * (self.probability_decrease) ** ind, "norm", np.linalg.norm(gradientk1))

                self.memory.add(np.concatenate([xk, xk1, [rew/max(abs(fk), 1e-6)]]))

                batch_size = self.batch_size
                if ((self.memory.depth) < self.batch_size):
                    batch_size = self.memory.depth
                mb = self.memory.extract_minibatch(batch_size)
                self.model.update(mb)
                singleton = []
                singleton.append(np.concatenate([xk1, [1]]))
                print("singleton",singleton)
                print(mb)
                self.generatormodel.update(singleton)
                xk = xk1
                fk = fk1
        if stopcondition == "notsatisfied":
            stopcondition = "maximumIterations " + str(iter)
            xstar = xk1
            fstar = fk1
        elif stopcondition == "notinitialized":
            stopcondition = "ERROR " + str(iter)
            xstar = xk1
            fstar = fk1

        return fstar, xstar, iter, stopcondition, episode

    def solve_with_gradients(self, x0, A, c):

        optimization_rate = self.starting_optimization_rate
        xstar = []
        fstar = 0
        iter = 0
        stopcondition = "notinitialized"

        for episode in range(self.episodes):
            self.updateseed(1)
            optimization_rate = self.starting_optimization_rate

            xk = []
            # if episode == 0:
            xk = np.array(x0)  +np.random.rand(self.n)*self.trust_region/2 - self.trust_region
            # else:
            # xk = xk1 + np.random.rand(self.n)-0.5

            gradientk = rl.nabla(xk, A, c)
            xk1 = xk.copy()
            fk = rl.f(xk, A, c)
            fk1 = rl.f(xk1, A, c)
            r = fk

            cwd = []
            wdmin = sys.float_info.max

            for step in range(self.step):
                # optimization_rate = self.starting_optimization_rate/(iter+1)
                ind = episode  # temporary choice
                iter += 1
                xk1 = self._select_action_with_gradients(xk, gradientk, fk, optimization_rate,
                                          self.initial_probability * (self.probability_decrease) ** ind)
                fk1 = rl.f(xk1, A, c)
                gradientk1 = rl.nabla(xk1, A, c)

                rew = np.linalg.norm(gradientk1) - np.linalg.norm(gradientk)  # - np.linalg.norm(gradientk1) #important if you have to maximize or minimize
                r += rew
                cwd.append(r)
                if step % self.watchdog == 0 and step > 1 and (self.initial_probability * ((self.probability_decrease) ** ind)) <= 0.05:
                    actminimum = np.min(np.array(cwd))
                    cwd = []
                    #print("------------watchdog activated-------------")
                    #print("actminimum", actminimum," wdmin", wdmin)

                    if actminimum >= wdmin and rew >= 0:
                        optimization_rate = optimization_rate/2.0
                        print("------------watchdog activatedpppppppppppppppppp-")
                    elif actminimum <= wdmin:
                        wdmin = actminimum

                print("ep", episode, " step", step, "function", fk1, "rew", rew, "r", r, "probability",
                      self.initial_probability * (self.probability_decrease) ** ind, "norm", np.linalg.norm(gradientk1))
                # prew = rew
                stopcondition = rl.stop(rl.nabla(xk1, A, c), xk, xk1, fk, fk1)
                if stopcondition != "notsatisfied":
                    xstar = xk1
                    fstar = fk1
                    print("STOP CONDITION", stopcondition)
                    break
                xs = xk1.copy()  # support value
                fs = fk1  # support value
                gradient_s = rl.nabla(xs,A,c)


                for future in range(self.experience):
                    xs1 = self._select_action_with_gradients(xs, gradient_s, fs,optimization_rate,
                                              self.initial_probability * (self.probability_decrease) ** (ind))
                    # rew = self.model.evaluate(np.concatenate([state, action, next_state]))
                    # print("----->",  np.concatenate[xs, xs1])
                    # fs = self.model.evaluate( np.concatenate([xs, xs1]))
                    gradient_s1 = rl.nabla(xs1, A, c)
                    fs1 = rl.f(xs1, A, c)
                    #gs = self.model.evaluate(np.concatenate([xs, xs1, gradient_s, gradient_s1, [fs], [fs1]]))
                    rew = rew + (self.future_relevance**(future+1)) * ( np.linalg.norm(gradient_s1)- np.linalg.norm(gradient_s))
                    #rew = rew + (self.future_relevance ** (future + 1)) * gs
                    xs = xs1
                    fs = fs1
                    gradient_s = gradient_s1



                # print("INSTANCE", np.concatenate([xk, xk1, [r]]))

                self.memory.add(np.concatenate([xk, xk1, gradientk, gradientk1, [fk], [fk1], [rew]]))

                batch_size = self.batch_size
                if ((self.memory.depth) < self.batch_size):
                    batch_size = self.memory.depth
                self.model.update(self.memory.extract_minibatch(batch_size))
                #self.model.longUpdate(self.memory.extract_minibatch(batch_size), batch_size, 4)
                xk = xk1
                fk = fk1
                gradientk = gradientk1
                # r = 0 #understand if this is valuable or not
        if stopcondition == "notsatisfied":
            stopcondition = "maximumIterations " + str(iter)
            xstar = xk1
            fstar = fk1
        elif stopcondition == "notinitialized":
            stopcondition = "ERROR " + str(iter)
            xstar = xk1
            fstar = fk1

        self.memory.print()

        return fstar, xstar, iter, stopcondition, episode






