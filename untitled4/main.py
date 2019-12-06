#from docplex.mp.model import Model
import numpy as np
#from toip import toip
from os import listdir
from os.path import isfile, join
import builder as bl

#mypath = "\micasaestucasa"
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
outputpath = "/home/giorgio/Desktop"
append = False

random_instance_size = 100
seed_list = np.random.randint(1, 1000, random_instance_size)
nobjectives = 3

#for f in onlyfiles:
 #   data = bl.read_data(f)
for seed in seed_list:
    data = bl.generate_data_linear(nobjectives,seed)
    cpmod, objs = bl.create_model( data )
    (solution, stats) = bl.apply_algorithm( cpmod, objs, gamma=1)
    bl.write_stats(outputpath, stats, append)
    if not append:
        append = True


