from docplex.mp.model import Model
import numpy as np
import time


class toip:
    def __init__(self, cpmod, obj1, obj2, obj3, gamma = 1):
        self.cpmod = cpmod
        self.objs = [obj1, obj2, obj3]
        self.gamma = gamma


    def _antiideal(self):
        y = []
        for obj in self.objs:
            self.cpmod.maximize(obj)
            sol = self.cpmod.solve(logOutput = False)
            y.append(sol.objective_value)
            self.cpmod.remove_objective();
        self.antiideal = y.copy()

    def _ideal(self):
        y = []
        for obj in self.objs:
            self.cpmod.minimize(obj)
            sol = self.cpmod.solve(logOutput = False)
            y.append(sol.objective_value)
            self.cpmod.remove_objective();
        self.ideal = y.copy()


    def _prod(w):
        ret = 1
        for wi in w:
            ret = ret*wi
        return ret

    def _weights(self, order, order_j):
        w = []
        w_j = []
        w.append(1.0)
        w_j.append(1.0)

        y = np.zeros((len(order), len(order)))

        self.cpmod.minimize(self.objs[order[0]])
        sol = self.cpmod.solve()
        y[order[0], order[0]]   = sol.objective_value
        y[order[0], order[1]] = self.objs[order[1]].solution_value
        y[order[0], order[2]] = self.objs[order[2]].solution_value
        self.cpmod.remove_objective()
        self.cpmod.minimize(self.objs[order[1]])
        sol = self.cpmod.solve()
        y[order[1], order[1]] = sol.objective_value
        y[order[1], order[0]] = self.objs[order[0]].solution_value
        y[order[1], order[2]] = self.objs[order[2]].solution_value
        self.cpmod.remove_objective()
        self.cpmod.minimize(self.objs[order[2]])
        sol = self.cpmod.solve()
        y[order[2], order[2]] = sol.objective_value
        y[order[2], order[1]] = self.objs[order[1]].solution_value
        y[order[2], order[0]] = self.objs[order[0]].solution_value
        self.cpmod.remove_objective()

        self.ideal = [y[j][j] for j in order]

        w23 = (self.gamma - 1e-8)/(y[order[1]][order[2]]-self.ideal[order[1]])
        w32 = (self.gamma - 1e-8)/(y[order[2]][order[1]]-self.ideal[order[2]])

        self.cpmod.minimize(self.objs[order[0]])
        sol = self.cpmod.solve()
        lam23 = (self.objs[order[1]] + self.objs[order[2]] * w23).solution_value
        lam32 = (self.objs[order[2]] + self.objs[order[1]] * w32).solution_value
        self.cpmod.remove_objective()
        self.cpmod.minimize(self.objs[order[1]] + self.objs[order[2]] * w23)
        sol = self.cpmod.solve()
        val23 = sol.objective_value
        self.cpmod.remove_objective()
        self.cpmod.minimize(self.objs[order[2]] + self.objs[order[1]] * w32)
        sol = self.cpmod.solve()
        val32 = sol.objective_value
        self.cpmod.remove_objective()


        lam23 = (self.gamma - 1e-8)/(lam23 - val23)
        lam32 = (self.gamma - 1e-8)/(lam32 - val32)

        w.append(lam23)
        w.append(lam23*w23)
        w_j.append(lam32)
        w_j.append(lam32*w32)


        wsup = []
        for i in range(len(order)):
            wsup.append(w[order[i]])
        wsup_j = []
        for i in range(len(order_j)):
            wsup_j.append(w_j[order_j[i]])
        return wsup, wsup_j

    def _are_equal(yj, y_j):
        p = sum( (abs(yj[i]-y_j[i])<= 1e-8) for i in  range(len(yj)))
        return p == len(y_j)


    def solve(self):
        #self._antiideal()
        #self._ideal()

        self.stats = []
        self.solution =  []

        start = time.time()

        order_j = [1, 0, 2]
        order = [0, 1, 2]
        w, w_j = self._weights(order, order_j)

        obj = self.cpmod.sum(self.objs[i] * w[i] for i in range(len(self.objs)))
        obj_j = self.cpmod.sum(self.objs[i] * w_j[i] for i in range(len(self.objs)))
        g23 = self.cpmod.sum(self.objs[order[i]] * w[order[i]]/w[order[1]] for i in range(1,len(self.objs)))

        problem_counter = 0
        conl = None
        while True:

            self.cpmod.minimize(obj_j)
            sol_j = self.cpmod.solve()
            problem_counter = problem_counter + 1
            if sol_j == None or sol_j.is_empty():
                self.stats.append('empty')
                self.stats.append('empty')
                self.stats.append('empty')
                return
            y_j = [self.objs[j].solution_value for j in range(len(self.objs))]
            self.cpmod.remove_objective()

            self.cpmod.minimize(obj)
            sol = self.cpmod.solve()
            problem_counter = problem_counter + 1
            yj = [self.objs[j].solution_value for j in range(len(self.objs))]

            g23val = g23.solution_value
            self.solution.append(yj)

            while not toip._are_equal(yj, y_j):
                conh = self.cpmod.add_constraint(self.objs[order[2]] <= (yj[order[2]] - self.gamma), ctname="conh")
                sol = self.cpmod.solve()
                problem_counter = problem_counter + 1
                print(problem_counter)
                yj = [self.objs[j].solution_value for j in range(len(self.objs))]
                self.solution.append(yj)
                self.cpmod.remove_constraint( conh)  # todo potrebbe dare errore vedere how it is exactly implemented (look for ct_arg)

            self.cpmod.remove_objective()
            self.cpmod.remove_constraint(conl)
            conl = self.cpmod.add_constraint(g23 <= (g23val - self.gamma), ctname="conl")

        self.stats.append((time.time()-start)/1000.0)
        self.stats.append(problem_counter)
        self.stats.append(len(self.solution))


    def get_solution_and_stats(self):
        return (self.solution, self.stats)