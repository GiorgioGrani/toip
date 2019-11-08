import numpy as np
import pandas as pd

class memory():
    def __init__(self, path = None):
        self.depth = 0
        if path == None:
            self.db = []
        else:
            self.db = self.compute_memory(path)
        #Derive the database from path or initialize an empty one
    def compute_memory(self, path = None):
        return []
    #Here is where the magic happens

    def add(self, array):
        self.db.append( array )
        self.depth = self.depth + 1

    def remove_old(self, expiration_date):
        self.db = self.db[ expiration_date : len(self.db)]
    def length(self):
        return len(self.db)
    def purge(self, windowlength):
        n = len(self.db)
        self.db = self.db[ n - windowlength : n]

    def extract_minibatch(self, size):

        array = np.random.choice( len(self.db), replace = False,  size = size)
        ret = []
        for i in  array:
            ret.append(self.db[i])
        return ret

    def print(self):
        np.savetxt("points.csv", np.asarray(self.db), delimiter=",")

