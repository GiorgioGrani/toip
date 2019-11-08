from memory import memory
#from model import model
from balerionmodel import balerionmodel
from time import time
import sys
import math
from docplex.mp.model import Model

import numpy as np
import keras


class balerion():

    def __init__(self, innermodel,
                 inneroptimizer,
                 outermodel,
                 outeroptimizer,
                 jobs,
                 machines,
                 sequences,
                 path=None,
                 M=100,
                 T=100,
                 peps=0.5,
                 delta=0.9,
                 gamma=0.5,
                 experience=1,
                 size=300,
                 randomized=True,
                 finalreward=1e5,
                 batch_size=10,
                 verbose=True,
                 npseed=None):

        if npseed != None:
            np.random.seed(npseed)

        self.innermodel = balerionmodel(innermodel, inneroptimizer)
        self.outermodel = balerionmodel( outermodel, outeroptimizer)
        self.memory = memory(path)
        self.T = T
        self.M = M
        self.peps = peps
        self.gamma = gamma
        self.delta = delta
        self.experience = experience
        if self.experience < 0:
            raise Exception("Experience must be a positive integer (experience > 0).")
        self.size = size
        self.randomized = randomized
        self.jobs = jobs
        self.machines = machines
        self.sequences = sequences
        self.finalreward = finalreward
        self.batch_size = batch_size
        self.verbose = verbose

        self.ndim = []
        n = 0
        for i in range(self.jobs):
            self.ndim.append(n)
            n = n + len(self.sequences[i])
        self.ndim.append(n)

        self.maximum_time = 0
        for i in range(self.jobs):
            for coup in self.sequences[i]:
                self.maximum_time = self.maximum_time + coup[1]

    def solve(self):

        best_solution = []
        best_timings = []
        best_reward = -(sys.float_info.max - 1)
        best_episode = -1
        retrewards = []
        makespan = 0
        makespanold = []
        makeslen = 0
        makesbest = (sys.float_info.max - 1)
        morerandom = -1
        best_schedule = []

        start = time()
        opt = False
        for episode in range(self.M):
            print("                                                   EPISODE",episode)
            exper = self.experience #min(episode + 1, self.experience)
            total_reward = 0
            solution = []
            timings = []
            timings.append(0)
            state = self.initializeState()
            schedule = []
            schedule.append(state)
            morerandom = morerandom + 1
            final = False
            activation = []
            for seq in self.sequences:
                activation.append(np.zeros(len(seq)))
            pivactivation = activation.copy()


            for i in range(self.T):
                xv = morerandom - 5
                ind = math.exp(xv)/(math.exp(xv)+ math.exp(-xv))  #morerandom # + i
                if ind >= 0.7:
                   ind = morerandom
                #ind = morerandom
                #print("                                                      move", i," ind", 1 * (self.delta ** (ind)))

                reward = 0


                next_reward, action, next_state, final, timing, activation, onehotria = self.selectAction(
                    1 * (self.delta ** (ind)),
                    state,
                    timing=True,
                    optimize=opt,
                    activation=activation, printable=False)
                solution.append(action)
                timings.append(timing)
                actual_reward = next_reward
                # print("  NS1", next_state)
                supstate = next_state

                total_reward = total_reward + actual_reward
                reward = next_reward

                pivactivation = []
                for v in activation:
                    pivactivation.append(v.copy())
                # print("ACTIVATION SOLVE", activation)

                if not final:
                    for j in range(exper):
                        next_reward, next_action, next_state, localfinal, pivactivation = self.selectAction(
                            1 * (self.delta ** (ind)),
                            next_state,
                            timing=False,
                            optimize=opt,
                            activation=pivactivation)
                        reward = reward + (self.gamma ** (j + 1)) * next_reward
                        if localfinal:
                            break
                # print("AFTER ACTIVATION SOLVE", activation)

                maxip = max(next_state)

                if maxip <= 1e9:
                    instance = np.concatenate([onehotria, state, action,next_state, [reward]])
                    self.memory.addlist(instance)
                state = supstate
                schedule.append(state)

                batch_size = self.batch_size
                if ((self.memory.depth) < self.batch_size):
                    batch_size = self.memory.depth

                mini_batch = self.memory.extract_minibatch(batch_size)

                self.outermodel.innerouterupdate(minibatch=mini_batch, inner_model = self.innermodel,
                                                 action_size_n= self.machines, state_size= len(state), kk= episode + i+1)

                if final:
                    # print("STATE FINAL", state)
                    break

            if best_reward <= total_reward:
                best_reward = total_reward
                best_episode = episode
                best_solution = solution
                best_schedule = schedule

                val = solution[len(solution) - 1]
                valmax = sys.float_info.max
                for i in val:
                    if i < valmax:
                        valmax = i
                timings.append(-valmax)
                best_timings = timings

            retrewards.append(total_reward)

            makespan = sum(t for t in timings)
            if makesbest > makespan:
                makesbest = makespan
            makespanold.append(makespan)
            makeslen = makeslen + 1
            wall = 10
            walle = wall + 1
            if makeslen > wall:
                thre = sum(makespanold[i] for i in range(makeslen - walle, makeslen - 1)) / wall
                var = sum((makespanold[i] - thre) ** 2 for i in range(makeslen - walle, makeslen - 1)) / wall
                std = np.sqrt(var)

                if makespan <= thre + 1 and makespan >= thre - 1:
                    #print(thre, " ", makespan, "  ", makesbest, "<-")
                    if makespan > makesbest + 5:
                        #print("MORE RANDOM!")
                        morerandom = 0
                    else:
                        break

            wl = 50000
            if self.memory.length() >= wl * 2:
                self.memory.purge(windowlength=wl)
            if self.verbose:
                if episode % 1 == 0:
                    print("\n\nEpisode: ", episode, ")   Reward: ",
                          total_reward,  # ")   Best solution: ", best_solution,
                          ")   Elapsed time: ", (time() - start), " s ", " STATE",
                          state)
                    print("Memory lenght", self.memory.length())

        if self.verbose:
            print("\n\nBest Episode: ", best_episode, ")   Best reward: ",
                  best_reward,  # ")   Best solution: ", best_solution,
                  ")   Elapsed time: ", (time() - start), " s")
            for i in range(len(best_schedule)):
                print("SCHEDULE[", i, "]", best_schedule[i])

        return best_episode, best_reward, retrewards, best_solution, (time() - start), best_timings

    def selectAction(self,  peps, state, timing=False,
                     optimize=False,  activation=[[]], printable=False):
        ''''''

        self.mincompletions = []
        for i in range(self.machines):
            self.mincompletions.append(1e10)
        self.minjobs = []
        for i in range(self.machines):
            self.minjobs.append((-1, -1))

        mach = state[(self.ndim[self.jobs]): (len(state))].copy()
        # print(" MACH",mach)
        # print((len(state)- self.machines) )
        # print( (len(state)) )
        # print(state)
        # print(self.ndim)
        setA = []
        for i in range(self.machines):
            if mach[i] <= 1e-9:
                setA.append(i)
        setR = []
        n = 0
        for i in range(self.jobs):
            j = -1
            for e in range(len(self.sequences[i])):
                if state[n + e] <= 1e-9:
                    jobactivitycheck = sum(activation[i][ac] for ac in range(len(activation[i])))
                    if jobactivitycheck <= 0.5:
                        j = e
                    break
            if j >= 0:
                machine = self.sequences[i][j][0]
                completion = self.sequences[i][j][1]
                if machine in setR:
                    if self.mincompletions[machine] > completion:
                        self.mincompletions[machine] = completion
                        self.minjobs[machine] = (i, j)
                else:
                    self.mincompletions[machine] = completion
                    self.minjobs[machine] = (i, j)
                    setR.append(machine)
            n = n + len(self.sequences[i])

        setA = np.array(setA)
        setR = np.array(setR)
        ria = np.intersect1d(setA, setR)
        onehotria = self.onehot(ria)
        #print(onehotria,"<------ONEHOTRIA")


        # minj = self.minjobs.copy()
        # minc = self.mincompletions.copy()
        # self.minjobs = []
        # self.mincompletions = []
        # for p in zip(minc, minj):
        #     if p[1] <= 1e9:
        #         self.minjobs.append(p[0])
        #         self.mincompletions.append(p[1])


        #print("------------------------> COMPLETIONS",self.mincompletions)
        #print("------------------------>  JOBS",self.minjobs)

        ''''''

        best_action = []
        best_reward = -(sys.float_info.max - 1)
        best_next_state = []
        best_activation = [[]]
        best_min = 0
        final = False
        rands = False
        check = True


        if (not (np.random.ranf() < peps)):
            check = False
            best_action = self.round(self.innermodel.multievaluate(np.concatenate([state, onehotria])))
            action_to_set = self.toSetFormat(best_action)
            act = np.array(action_to_set)

            intersec = np.intersect1d(ria, act)
            if len(intersec) <= 0:
                check = True

            # print("GOTCHA", ria,"\nACTION_TO_SET", action_to_set,"\nBEST_ACTION", best_action)
            best_next_state, final, best_min, best_activation = self.computestep(state, action_to_set, activation,
                                                                                 printable)

            # print([state, best_action, best_next_state])

            # best_action = self.onehot(best_action)


        elif check:  # if action_set != [[]]:
            rands = True
            best_action = self.random_action(ria)
            #print("RIAAAAAAAAAAAAAAAAAA", ria)
            best_next_state, final, best_min, best_activation = self.computestep(state, best_action, activation,
                                                                                 printable)
            #best_action = self.onehot(best_action)
            best_action = self.one(best_action)
            #self.innermodel.train(np.concatenate([state, best_action]))#for now we decide not to train the inner model on such instance
            # best_reward = self.model.evaluate(np.concatenate([state, best_action, best_next_state]))
        best_reward = self.outermodel.evaluate(np.concatenate([onehotria,state, best_action, best_next_state]))

        if timing:# or (rands and peps >= 0.2):
            best_reward = -best_min
        #print("BEST_REWARD",best_reward,"timing", timing)
        if timing:
            return best_reward, best_action, best_next_state, final, best_min, best_activation, onehotria#, self.onehot(ria)
        return best_reward, best_action, best_next_state, final, best_activation

    def round(self, vec):
        return [ 1 if round(vec[i])>=0.5 else 0 for i in range(len(vec))]


    def toSetFormat(self, action):
        ind = [i if action[i] > 0.5 else -2 for i in range(len(action))]
        return [ind[i] for i in range(len(ind)) if ind[i] >= -0.5]


    def random_action(self, ria):
        n = len(ria)
        # print("n",n," ria[0]", ria[np.random.randint(0, len(ria))])
        if n > 0:
            size = np.random.randint(1, n + 1)
            # print("size",size)
            ind = np.random.choice(ria, replace=False, size=size)
            ind = np.sort(ind)
            return ind
        else:
            ret = []
            return ret

    def initializeState(self):
        n = self.ndim[self.jobs]
        state = np.zeros(n + self.machines)

        return state

    def onehot(self, action):
        ret = np.zeros(self.machines)
        for i in action:
            ret[i] = self.mincompletions[i]
        return ret

    def one(self, action):
        ret = np.zeros(self.machines)
        for i in action:
            ret[i] = 1
        return ret

    def computestep(self, state, action, activation, printable=False):
        #print("COMPUTE STEP ACTION", action)
        ###transitional state###

        #print("_________________________INSIDE COMPUTE STEP_______________________")
        # print("________________STATE ", state)
        # print("_______________ACTION ", action)
        act = activation.copy()

        newstate = state.copy()
        mach = state[(len(state) - self.machines): (len(state))].copy()
        updatable_jobs = []

        bfinal = False

        for elem in action:
            mach[elem] = self.mincompletions[elem]
            if mach[elem] >= 1e9:
                bfinal = True
            jobi = self.minjobs[elem][0]
            updatable_jobs.append(self.minjobs[elem])
            for operationj in range(len(self.sequences[jobi])):
                if self.sequences[jobi][operationj][0] == elem:
                    act[jobi][operationj] = 1
                    break  # there is no repetition

        for i in updatable_jobs:
            job = i[0]
            assignment = i[1]
            pos = self.ndim[job]
            newstate[pos + assignment] = self.sequences[job][assignment][1]

        # print("_______________NEWSTATE ", newstate)

        ###next state###
        min = 1e10
        flag = False
        for machine in range(len(mach)):
            if mach[machine] < min and mach[machine] >= 1e-9:
                min = mach[machine]
                flag = True
        if not flag:
            min = 0
        for machine in range(len(mach)):
            if mach[machine] >= 1e-9:
                mach[machine] = mach[machine] - min
            if mach[machine] <= 1e-9:  # freeing phase
                for jobi in range(len(self.sequences)):
                    for operationj in range(len(self.sequences[jobi])):
                        if self.sequences[jobi][operationj][0] == machine:
                            act[jobi][operationj] = 0

        newstate[(len(state) - self.machines): (len(state))] = mach
        # this is redundant but more clear in my thoughts

        ###check final###
        final = True
        for i in range(self.ndim[self.jobs]):
            if state[i] <= 1e-9:
                final = False
                break
        if bfinal:
            final = True
            min = self.finalreward
        if printable:
            print("_______________OLD _STATE ", state)
            print("_______________ACTION     ", action)
            print("_______________NEXT_STATE ", newstate)
            print("_______________ACT        ", act)
            print("\n")

        return newstate, final, min, act



