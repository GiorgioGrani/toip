from memory import memory
from model import model
from time import time
import sys
from docplex.mp.model import Model

import numpy as np
import keras

class train():
    def __init__(self, Qmodel,
                 optimizer,
                 jobs,
                 machines,
                 sequences,
                 path = None,
                 M = 100,
                 T = 100,
                 peps = 0.5,
                 delta = 0.9,
                 gamma = 0.5,
                 experience = 1,
                 size = 300,
                 randomized = True,
                 finalreward = 1e5,
                 batch_size = 10,
                 verbose = True,
                 npseed = None):
        if npseed != None :
            np.random.seed(npseed)

        self.model = model(Qmodel, optimizer)
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
        self.ndim.append( n )

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
            exper = min( episode + 1, self.experience)
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

                ind = morerandom # + i

                reward = 0

                action_set, ria = self.action_set(state, self.size,self.randomized, optimize = opt, activation = activation)

                next_reward, action, next_state, final, timing, activation = self.selectAction(
                                                         action_set,
                                                         self.peps*(self.delta**(ind)),
                                                         state,
                                                         timing = True,
                                                         optimize = opt,
                                                         ria = ria, activation = activation, printable = False)
                solution.append(action)
                timings.append(timing)
                actual_reward = next_reward
                #print("  NS1", next_state)
                supstate = next_state

                total_reward = total_reward + actual_reward
                reward = next_reward

                pivactivation = []
                for v in activation:
                    pivactivation.append(v.copy())
                #print("ACTIVATION SOLVE", activation)

                if not final:
                    for j in range(exper):
                        action_set, ria = self.action_set(next_state,
                                                          self.size, self.randomized,
                                                          optimize = opt, activation = pivactivation)
                        next_reward, next_action, next_state, localfinal, pivactivation = self.selectAction(
                            action_set,
                            self.peps * (self.delta ** (ind )),
                            next_state,
                            timing = False,
                            optimize = opt,
                            ria = ria, activation = pivactivation)
                        reward = reward + (self.gamma ** (j+1)) * next_reward
                        if localfinal:
                            break
                #print("AFTER ACTIVATION SOLVE", activation)

                instance = np.concatenate([ state, action, next_state, [reward] ])
                self.memory.add(instance)
                state = supstate
                schedule.append(state)



                batch_size = self.batch_size
                if( (self.memory.depth) < self.batch_size):
                    batch_size = self.memory.depth

                mini_batch = self.memory.extract_minibatch(batch_size)
                #siz = round(batch_size/2)
                #if siz <= 0:
                #    siz = 1
                #self.model.longUpdate( mini_batch, siz, 1, True)
                self.model.update(minibatch = mini_batch)

                if final:
                    #print("STATE FINAL", state)
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
                thre = sum(makespanold[i] for i in range(makeslen - walle, makeslen - 1))/wall
                var = sum((makespanold[i] - thre)**2 for i in range(makeslen - walle, makeslen - 1))/wall
                std = np.sqrt(var)


                #print("thre", thre, "std",std,"ref", makespan," thre+.05std", thre + 0.01*std)
                #print(makespanold)
                if makespan <= thre + 1 and makespan >= thre - 1:
                    print(thre," ",makespan,"  ", makesbest,"<-")
                    if makespan > makesbest + 5:
                        print("MORE RANDOM!")
                        morerandom = 0
                    else:
                        break

            wl = 500
            if self.memory.length() >= wl*2:
                self.memory.purge(windowlength=wl)
            if self.verbose:
                if episode%1 == 0:

                    print("\n\nEpisode: ", episode, ")   Reward: ",
                      total_reward,  # ")   Best solution: ", best_solution,
                      ")   Elapsed time: ", (time() - start), " s "," STATE",
                          state)
                    print("Memory lenght", self.memory.length())




        if self.verbose:
            print("\n\nBest Episode: ", best_episode, ")   Best reward: ",
              best_reward, #")   Best solution: ", best_solution,
              ")   Elapsed time: ", (time() - start), " s")
            for i in range(len(best_schedule)):
                print("SCHEDULE[", i, "]", best_schedule[i])

        return best_episode, best_reward, retrewards,  best_solution, (time() - start), best_timings
    '''
    def play(self):

        best_solution = []
        best_timings = []
        best_reward = -(sys.float_info.max - 1)
        best_episode = -1
        retrewards = []

        start = time()
        for episode in range(1):
            total_reward = 0
            solution = []
            timings = []
            timings.append(0)
            state = self.initializeState()

            final = False
            opt = False
            for i in range(self.T):

                ind = self.M + i

                reward = 0
                action_set, ria = self.action_set(state,
                                                  self.size,
                                                  self.randomized,
                                                  optimize=opt)

                next_reward, action, next_state, final, timing = self.selectAction(
                    action_set,
                    self.peps * (self.delta ** (ind)),
                    state, timing=True, optimize = opt, ria = ria)
                solution.append(action)
                timings.append(timing)
                actual_reward = next_reward
                # print("  NS1", next_state)
                supstate = next_state

                total_reward = total_reward + actual_reward
                reward = next_reward
                if final:
                    break


                state = supstate

            if best_reward <= total_reward:
                best_reward = total_reward
                best_episode = episode
                best_solution = solution

                val = solution[len(solution) - 1]
                valmax = sys.float_info.max
                for i in val:
                    if i < valmax:
                        valmax = i
                timings.append(-valmax)
                best_timings = timings

            retrewards.append(total_reward)

            if self.verbose:
                print("\n\nEpisode: ", episode, ")   Reward: ",
                      total_reward,  # ")   Best solution: ", best_solution,
                      ")   Elapsed time: ", (time() - start), " s ")

        if self.verbose:
            print("\n\nBest Episode: ", best_episode, ")   Best reward: ",
                  best_reward,  # ")   Best solution: ", best_solution,
                  ")   Elapsed time: ", (time() - start), " s ")
        return best_episode, best_reward, retrewards, best_solution, (time() - start), best_timings
    '''

    def selectAction( self, action_set , peps, state, timing = False,
                      optimize = False, ria = None, activation = [[]], printable = False):
        best_action = []
        best_reward = -(sys.float_info.max - 1)
        best_next_state = []
        best_activation = [[]]
        best_min = 0
        final = False
        rands = False
        if np.random.ranf() < peps:
            rands = True
            if not optimize:
                #if len(action_set)
                best_action = action_set[np.random.randint(0, len(action_set))]
            else:
                #best_action = action_set[np.random.randint(0, len(action_set))]
                #print("RIA",ria, "STATE",state)
                best_action = self.random_action(ria)
            best_next_state,  final, best_min, best_activation = self.computestep(state, best_action, activation, printable)
            best_action = self.onehot(best_action)
            #best_reward = self.model.evaluate(np.concatenate([state, best_action, best_next_state]))
        else :#if action_set != [[]]:
            for action in action_set:
                next_state, local_final, min, local_activation= self.computestep(state, action, activation, printable)
                action = self.onehot(action)
                rew = self.model.evaluate(np.concatenate([state, action, next_state]))
                #if rew <= -1e6:
                #    rew = -1e5
                #print("rew", rew, ">", best_reward," ->", str(rew>best_reward))
                if rew > best_reward:
                    best_reward = rew
                    best_action = action
                    best_next_state = next_state
                    best_min = min
                    final = local_final
                    best_activation = local_activation
        #else:
            #print("_________________________________________________________________DEBUG HAPPENED")
          #  mach = state[ (self.ndim[self.jobs]): (len(state))].copy()
           # max = 0
            #for m in mach:
             #   if m > max:
              #      max = m
            #best_min = max
            #final = True
            #next_state, local_final, min = self.computestep(state, best_action)
            #best_action = self.onehot(best_action)
            #best_next_state = next_state

        #best_reward = self.trueReward( final, best_next_state ) #- best_min
        #if timing or rands:
        best_reward = -best_min
        #if rands:
            #print("ria",ria," best_action", best_action, " rands", rands, "action_set", action_set)
        if timing:
            return best_reward, best_action, best_next_state, final, best_min, best_activation
        return best_reward, best_action, best_next_state, final, best_activation

    def random_action(self, ria):
        n = len(ria)
        #print("n",n," ria[0]", ria[np.random.randint(0, len(ria))])
        if n > 0:
            size = np.random.randint(1,n+1)
            #print("size",size)
            ind = np.random.choice(ria, replace=False,size=size)
            ind = np.sort(ind)
            return ind
        else:
            ret = []
            return ret





    '''
    def trueReward(self, final, nextstate):
        if final:
            return self.finalreward
        rew = 0
        for i in range(0, self.jobs):
            start = self.ndim[i]
            end = self.ndim[i + 1]
            for j in range(start, end):
                # print("j",j,"__", len(nextstate))
                if nextstate[j] <= 1e-9:
                    # print("j",j,"  nextstatedimension ", len(nextstate))
                    # print("i",i," j",j, "  j-start ", j-start)
                    # print("start ", start, "  end ", end)
                    # print( self.sequences[i])
                    # print( self.sequences[i][j-start])
                    # print( self.sequences[i][j-start][1])
                    rew = rew + self.sequences[i][j - start][1]
        return rew
    '''


    def action_set(self, state, size=-1, randomized = False, optimize = False, activation = [[]]): #todo implement size and randomized

        self.mincompletions = []
        for i in range(self.machines):
            self.mincompletions.append(1e10)
        self.minjobs = []
        for i in range(self.machines):
            self.minjobs.append((-1,-1))

        mach = state[ (self.ndim[self.jobs]): (len(state))].copy()
        #print(" MACH",mach)
        #print((len(state)- self.machines) )
        #print( (len(state)) )
        #print(state)
        #print(self.ndim)
        setA = []
        for i in range(self.machines):
            if mach[i] <= 1e-9:
                setA.append(i)
        setR = []
        n = 0
        for i in range(self.jobs):
            j = -1
            for e in range(len(self.sequences[i])):
                if state[ n + e ] <= 1e-9:
                    jobactivitycheck = sum(activation[i][ac] for ac in range( len( activation[i])))
                    if jobactivitycheck <= 0.5:
                        j = e
                    break
            if j >= 0 :
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

        result = []

        if optimize:
            result = self.optimize_action(ria, state)
        elif size <= 0:
            result = self.powerSet(ria, -1, [])
        elif not randomized:
            self.globalsize = 0
            self.breakPowerSet = False
            result = self.powerSetSized(ria, -1 , [], size)
        else:
            self.globalsize = 0
            self.breakPowerSet = False
            result = self.powerSetSizedAndRandomized(ria, -1, [], size)

        #print(".................RESULT.................")
        #print(result)
        #print("........................................")

        return result, ria


    '''

    def optimize_action(self, ria, state):
        cplexmodel = Model()
        layers = self.model.model.layers
        weights = []
        count = -1
        for layer in layers:
            count = count+1
            w = layer.get_weights()
            w0 = w[0]
            w1 = w[1]
            #print(w0.shape,"   ", w1.shape)
            ws = []
            for j in range(len(w1)):
                ws0 = []
                for i in range(len(w0)):
                    ws0.append(w0[i,j])
                ws0.append(w1[j])
                ws.append(ws0)
            print("layer", count, " w" , ws)
            weights.append(ws)

        z = cplexmodel.binary_var_list(range(self.machines), name = "z")
        for i in ria:
            cplexmodel.add_constraint(z[i] == 0)
        q = []
        q.append(z)
        for l in range(len(layers)-1):
            q.append(
                cplexmodel.continuous_var_list(
                    range(len(weights[l])),
                    lb = 0,
                    name = "q"+str(l+1)+"_" )  )
            print("q[l]",q[l+1])
            
            
            
            
            
            
        for l in range(2, len(layers)):
            for i in range(len(q[l])):
                cplexmodel.add_constraint(
                    cplexmodel.sum(weights[l-1][i][j]*q[l-1][j] for j in range(len(q[l-1])))  + weights[l-1][i][len(weights[l-1][i])-1]  <= q[l][i]
                                   )

        n = len(layers)-1
        print(len(q[n]),",,,",n,"...__...__ ",weights[n][0])
        #cplexmodel.minimize(cplexmodel.sum(weights[n][0][j] * q[n][j] for j in range(len(q[n]))))
        cplexmodel.minimize(cplexmodel.sum(100*weights[n][0][j] * q[n][j] for j in range(len(q[n]))))

        #([state, action, next_state, [reward]])

        w0 = weights[0]
        q0 = q[1]
        for i in range(len(q0)):
            cplexmodel.add_constraint(
                cplexmodel.sum(w0[i][j] for j in range(len(state)))+
                cplexmodel.sum(w0[i][len(state)+j]*z[j]*self.mincompletions[j] for j in range(self.machines))+
                cplexmodel.sum(w0[i][ j] for j in range( len(state)+self.machines, len(w0[i])))
                <= q0[i]
            )

        #no null solution
        cplexmodel.add_constraint(            cplexmodel.sum(z[i] for i in range(len(z))) >= 1        )

        cplexmodel.solve(log_output = True)
        print(cplexmodel.export_as_lp_string())
        ret = []
        result = []
        for i in range(len(z)):#portare la soluzione da onehot a normale con i mincompletions.
            if z[i].solution_value>= 0.5:
                ret.append(i)
        result.append(ret)
        print("RESulTTTTTTT   TTTT   TTTT   TTTT    TTTT    TTTT    TTTT", result)
        return result
    '''

    def optimize_action(self,ria, state):
        cplexmodel = Model()
        layers = self.model.model.layers
        weights = []
        count = -1
        for layer in layers:
            count = count+1
            w = layer.get_weights()
            w0 = w[0]
            w1 = w[1]
            #print(w0.shape,"   ", w1.shape)
            ws = []
            for j in range(len(w1)):
                ws0 = []
                for i in range(len(w0)):
                    ws0.append(w0[i,j])
                ws0.append(w1[j])
                ws.append(ws0)
            #print("layer", count, " w" , ws)
            weights.append(ws)

        z = cplexmodel.binary_var_list(range(self.machines), name = "z")

        nmac = np.array(range(self.machines))
        nria = np.setdiff1d(nmac, ria)
        for i in nria:
            cplexmodel.add_constraint(z[i] == 0)
        q = []
        gamma = []
        gamma.append([0])
        q.append(z)
        for l in range(len(layers)-1):
            q.append(
                cplexmodel.continuous_var_list(
                    range(len(weights[l])),
                    lb = 0,
                    name = "q_"+str(l+1)+"_" )  )
            gamma.append(
                cplexmodel.binary_var_list(
                    range(len(weights[l])),
                    name = "gamma_"+str(l+1)+"_" )   )

        ######OBJECTIVE
        n = len(layers)-1
        #print(len(q[n]),",,,",n,"...__...__ ",weights[n][0])
        #cplexmodel.minimize(cplexmodel.sum(weights[n][0][j] * q[n][j] for j in range(len(q[n]))))
        cplexmodel.minimize(cplexmodel.sum(100*weights[n][0][j] * q[n][j] for j in range(len(q[n]))))
        bigM = 1e6


        ######CONSTRAINTS AND OTHER VARIABLES
        #for l in range(2, len(layers)):
        for l in range(2, len(layers)):
            for i in range(len(q[l])):



                #q constraints
                cplexmodel.add_constraint(
                    cplexmodel.sum(weights[l - 1][i][j] * q[l - 1][j] for j in range(len(q[l - 1]))) +
                    weights[l - 1][i][len(weights[l - 1][i]) - 1] <= q[l][i] + bigM * (1.0 - gamma[l][i])
                )
                cplexmodel.add_constraint(
                    cplexmodel.sum(weights[l - 1][i][j] * q[l - 1][j] for j in range(len(q[l - 1]))) +
                    weights[l - 1][i][len(weights[l - 1][i]) - 1] >= q[l][i] - bigM * (1.0 - gamma[l][i])
                )
                cplexmodel.add_constraint(
                    q[l][i] <= bigM * gamma[l][i]
                )
                cplexmodel.add_constraint(
                    q[l][i] >= - bigM * gamma[l][i]
                )

                #gamma constraints

                cplexmodel.add_constraint(
                    cplexmodel.sum(weights[l - 1][i][j] * q[l - 1][j] for j in range(len(q[l - 1]))) +
                    weights[l - 1][i][len(weights[l - 1][i]) - 1] <=  bigM * (gamma[l][i])
                )
                cplexmodel.add_constraint(
                    cplexmodel.sum(weights[l - 1][i][j] * q[l - 1][j] for j in range(len(q[l - 1]))) +
                    weights[l - 1][i][len(weights[l - 1][i]) - 1] >=  - bigM * (1.0 - gamma[l][i])
                )

        #special first contrAINTS
        w0 = weights[0]
        q0 = q[1]
        gamma0 = gamma[1]
        for i in range(len(q0)):
            #print("______________DEBUG DEBUG q0[i]",q0[i], " q0", q0)
            cplexmodel.add_constraint(
                cplexmodel.sum(w0[i][j] for j in range(len(state))) +
                cplexmodel.sum(w0[i][len(state) + j] * z[j] * self.mincompletions[j] for j in ria) +
                cplexmodel.sum(w0[i][j] for j in range(len(state) + self.machines, len(w0[i])))
                <= q0[i] + bigM * (1.0 - gamma0[i])
            )
            cplexmodel.add_constraint(
                cplexmodel.sum(w0[i][j] for j in range(len(state))) +
                cplexmodel.sum(w0[i][len(state) + j] * z[j] * self.mincompletions[j] for j in ria) +
                cplexmodel.sum(w0[i][j] for j in range(len(state) + self.machines, len(w0[i])))
                >= q0[i] - bigM * (1.0 - gamma0[i])
            )


            cplexmodel.add_constraint(
                q0[i] <= bigM * gamma0[i]
            )
            cplexmodel.add_constraint(
                q0[i] >= - bigM * gamma0[i]
            )

            cplexmodel.add_constraint(
                cplexmodel.sum(w0[i][j] for j in range(len(state))) +
                cplexmodel.sum(w0[i][len(state) + j] * z[j] * self.mincompletions[j] for j in ria) +
                cplexmodel.sum(w0[i][j] for j in range(len(state) + self.machines, len(w0[i])))
                <=  bigM *  gamma0[i]
            )

            cplexmodel.add_constraint(
                cplexmodel.sum(w0[i][j] for j in range(len(state))) +
                cplexmodel.sum(w0[i][len(state) + j] * z[j] * self.mincompletions[j] for j in ria) +
                cplexmodel.sum(w0[i][j] for j in range(len(state) + self.machines, len(w0[i])))
                >=  - bigM * (1.0 - gamma0[i])
            )

            # no null solution
        cplexmodel.add_constraint(cplexmodel.sum(z[i] for i in range(len(z))) >= 1)

            #start = time()
        cplexmodel.set_time_limit(60)
        check = cplexmodel.solve(log_output=False)
            #self.elaps = time() - start
        #print("CHECK",check)
            #print("RIA__",ria)
            #print("STATE__",state)
        #print(cplexmodel.export_as_lp_string())
            #for p in self.mincompletions:
               # print("mincolpetions",p)
        ret = []
        result = []
        if check == None:
            result.append(ret)
            #print("----------------debug RIA", ria)
            return result
        for i in range(len(z)):  # portare la soluzione da onehot a normale con i mincompletions.
            if z[i].solution_value >= 0.5:
                ret.append(i)
        result.append(ret)
        #print("RESulTTTTTTT   TTTT   TTTT   TTTT    TTTT    TTTT    TTTT", result,"   riaoptset", ria)
        return result


    def powerSet(self, ria, i, action):
        if len(ria) == i + 1:
            piv = action.copy()
            return [piv]
        ret = []
        if len(action) > 0:
            ret.append(action)
        for j in range(i + 1, len(ria)):
            piv = action.copy()
            piv.append(ria[j])
            new = self.powerSet(ria, j, piv)
            for elem in new:
                ret.append(elem)
        return ret

    def powerSetSized(self, ria, i, action, size):
        if self.breakPowerSet:
            return
        if len(ria) == i + 1:
            piv = action.copy()
            return [piv]
        ret = []
        if len(action) > 0:
            ret.append(action)
        for j in range(i + 1, len(ria)):
            piv = action.copy()
            piv.append(ria[j])
            new = self.powerSetSized(ria, j, piv, size)
            self.globalsize = self.globalsize + len(new)
            gap = self.globalsize - size
            if gap < 0:
                for elem in new:
                    ret.append(elem)
            else:
                for ielem in range(min(gap, len(new))):
                    ret.append(new[ielem])

                self.breakPowerSet = True
        return ret

    def powerSetSizedAndRandomized(self, ria, i, action, size):
        if self.breakPowerSet:
            return
        if len(ria) == i + 1:
            piv = action.copy()
            self.globalsize = self.globalsize + 1
            return [piv]
        ret = []
        if len(action) > 0:
            ret.append(action)
        for j in range(i + 1, len(ria)):
            piv = action.copy()
            piv.append(ria[j])
            new = self.powerSetSizedAndRandomized(ria, j, piv, size)
            if new != None :
                ins = np.random.randint(0,len(new))+1
                self.globalsize = self.globalsize + ins
                gap = self.globalsize - size
                indeces = []
                if gap < 0:
                    indeces = np.random.choice(range(len(new)), replace=False, size = ins)
                else:
                    if gap != 0:
                        indeces = np.random.choice(range( len(new)), replace=False, size=min(gap,ins))
                    self.breakPowerSet = True
                indeces.sort()
                for ielem in indeces:
                    ret.append(new[ielem])
                #print("i",i," globalsize", self.globalsize," break", self.breakPowerSet," gap", gap, " ret", ret,
                   #   "len(new)", len(new), "indeces", indeces)
        if ret == []:
            ret = [[]]
        return ret


    # print( powerSet(['a','b','c','d','e'], -1, []))


    def initializeState(self):
        n = self.ndim[self.jobs]
        state = np.zeros(n + self.machines)

        return state

    def onehot(self, action):
        ret = np.zeros(self.machines)
        for i in action:
            ret[i] = self.mincompletions[i]
        return ret


    def computestep(self, state, action, activation, printable = False):
        #print("COMPUTE STEP ACTION", action)
        ###transitional state###

        #print("_________________________INSIDE COMPUTE STEP_______________________")
        #print("________________STATE ", state)
        #print("_______________ACTION ", action)
        act = activation.copy()

        newstate = state.copy()
        mach = state[(len(state) - self.machines): (len(state) )].copy()
        updatable_jobs = []

        for elem in action:
            mach[elem] = self.mincompletions[elem]
            jobi = self.minjobs[elem][0]
            updatable_jobs.append(self.minjobs[elem])
            for operationj in range(len(self.sequences[jobi])):
                if self.sequences[jobi][operationj][0] == elem:
                    act[jobi][operationj] = 1
                    break #there is no repetition

        for i in updatable_jobs:
            job = i[0]
            assignment = i[1]
            pos = self.ndim[job]
            newstate[pos + assignment] = self.sequences[job][assignment][1]

        #print("_______________NEWSTATE ", newstate)




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
            if mach[machine] >=1e-9 :
                mach[machine] = mach[machine] - min
            if mach[machine] <= 1e-9: #freeing phase
                for jobi in range(len(self.sequences)):
                    for operationj in range( len(self.sequences[jobi])):
                        if self.sequences[jobi][operationj][0] == machine:
                            act[jobi][operationj] = 0



        newstate[(len(state) - self.machines): (len(state) )] = mach
        #this is redundant but more clear in my thoughts

        ###check final###
        final = True
        for i in range(self.ndim[self.jobs]):
            if state[i] <=1e-9:
                final = False
                break
        if printable:
            print("_______________OLD _STATE ", state)
            print("_______________ACTION     ", action)
            print("_______________NEXT_STATE ", newstate)
            print("_______________ACT        ",act)
            print("\n")



        return newstate, final, min, act





