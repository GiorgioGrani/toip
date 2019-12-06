from docplex.mp.model import Model
import numpy as np
from toip import toip
from os import listdir
from os.path import isfile, join

def read_data(file):
    return 1
def create_model(data):
    (C, A, b, lb, ub) = data
    cpmod = Model("TOIP")
    x = [cpmod.integer_var(lb=lb[i], ub=ub[i], name="x"+str(i)) for i in range(len(C[0]))]
    objs =[cpmod.sum( C[i][j]*x[j]for j in range(len(C[0]))) for i in range(len(C))]
    for i in range(len(A)):
        cpmod.add_constraint(cpmod.sum( A[i][j]*x[j] for j in range(len(C[0]))) >= b[i])
    return cpmod, objs



    return 1
def apply_algorithm(cpmod, objs, gamma = 1):
    toipsolver = toip(cpmod, objs[0], objs[1], objs[2], gamma)
    toipsolver.solve()
    return toipsolver.get_solution_and_stats()
def write_stats(outputpath, stats, append):
    flag = 'w'
    if append:
        flag = 'a'

    outfile = open(outputpath + "/stats.csv", flag)

    if append:
        outfile.write( "\'TOIP_time\', \'TOIP_integer_problems\',\'TOIP_number_of_solutions\'\n")
    s = str(stats[0])+", "+str(stats[1])+", "+str(stats[2])
    outfile.write(s)
    outfile.close()


def generate_data_linear(nobjectives, seed):
    np.random.seed(seed)
    nvars = np.random.randint(10,30)
    if nvars%2 !=0:
        nvars = nvars +1
    #print(round(nvars/2))
    C = np.random.randint(-2, 2*nvars, (nobjectives, nvars) )
    A = np.random.randint(0, 10, (round(nvars/2), nvars))
    suma = [sum(A[i][j] for j in range(nvars)) for i in range(round(nvars/2))]
    maxsum = np.max(suma)
    lb = np.zeros(nvars)
    ub = [np.random.randint(2 * maxsum, 3 * maxsum) for i in range(nvars)]
    b =  [np.random.randint(0, suma[i]) for i in range(round(nvars/2))]

    data = (C, A, b, lb, ub)
    return data





