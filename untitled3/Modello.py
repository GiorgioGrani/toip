from docplex.mp.model import Model

class Modello:
    def __init__(self, p = None, w = None, c = None):
        self.cpmod = Model( name = "Esempio")
        self.n = 0
        self.build(p, w, c)

    def build(self, p = None, w = None, c = None):
        if p != None and w != None and c != None:
            self.n = len(p)
            self.vars = self.variables( self.n)
            self.con = self.constraint(lhs = w, rhs = c)
            self.obj = self.objective( p )

    def variables(self, n):
        varlist = self.cpmod.binary_var_list( n, name = 'x')
        return varlist

    def constraint(self, lhs, rhs):
        left = self.cpmod.sum(  lhs[i]*self.vars[i] for i in range(self.n))
        con = self.cpmod.add_constraint( left <= rhs, ctname = "knapsack")
        return con

    def objective(self, p):
        obj = self.cpmod.sum(p[i]*self.vars[i] for i in range(self.n))
        sense = self.cpmod.maximize( obj )
        return sense

    def solve(self):
        self.cpmod.solve()

    def printsol(self):
        self.cpmod.print_solution()

    def printmodel(self):
        string = self.cpmod.export_as_lp_string()
        print(string)
