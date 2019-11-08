
from Modello import Modello

#initialize n
n = 4

#initialize params
p = [30, 36, 45]
w = [0.1, 0.2, 0.3]
c = 0.4

#solvemodel
modelo = Modello( p = p, w = w, c = c)
modelo.solve()
modelo.printsol()
modelo.printmodel()