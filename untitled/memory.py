import numpy as np

class memory():
    def __init__(self, path = None):
        self.pathflag = False
        self.depth = 0
        if path == None:
            self.db = None
        else:
            self.pathflag = True
            self.db = self.compute_memory(path)
        #Derive the database from path or initialize an empty one
    def compute_memory(self, path = None):
        return []
    #Here is where the magic happens

    def add(self, array):
        if not self.pathflag:
            self.db = np.matrix([array])
            self.pathflag = True
            self.depth = 1
        else:
            self.db = np.append(self.db,  [array] , axis = 0)
            self.depth = self.depth + 1

    def addlist(self, array):
        if not self.pathflag:
            self.db = []#np.matrix([array])
            self.db.append(array)
            self.pathflag = True
            self.depth = 1
        else:
            #self.db = np.append(self.db, [array], axis=0)
            self.db.append(array)
            self.depth = self.depth + 1
        #print("DB\n",self.db)
    def remove_old(self, expiration_date):
        self.db = self.db[ expiration_date : len(self.db)]
    def length(self):
        return len(self.db)
    def purge(self, windowlength):
        n = len(self.db)
        self.db = self.db[ n - windowlength : n]

    def extract_minibatch(self, size):
        #print("ssssssssssssssssssssssssssssssssssSIZE ", size)
        #print("MEMORY",self.db)

        array = np.random.choice( len(self.db), replace = False,  size = size)
        ret = []
        for i in  array:
            ret.append(self.db[i])
        return ret

