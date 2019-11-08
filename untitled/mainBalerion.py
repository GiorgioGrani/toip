import numpy as np
import keras as ks
from Balerion import balerion
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
#from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import spline
from docplex.mp.model import Model
from time import time



def createInstances( jobs, machines):
    sequences = []
    length = np.random.choice(machines, replace=True, size=jobs)
    n = 0
    for i in range(len(length)):
        if length[i] <= machines/2:
            length[i] = machines
        n = n + length[i]
    for i in range(jobs):
        array = np.random.choice(machines, replace=False, size=length[i])
        comp = np.random.randint(low = 1, high = jobs*10, size = machines)
        sequence = []
        for s, t in zip(array, comp):
            add = t
            if add < 1:
                add = 1
            sequence.append((s, add))
        sequences.append(sequence)
    print("JOBS: ", jobs)
    print("MACHINES: ", machines)
    for i in range(jobs):
        print("Job N°", i, " : ", sequences[i])
    return sequences, n

def createModel( layers, instance_dimension ):
    model = Sequential()


    tup = layers[0]
    model.add(Dense(tup[0], input_dim=instance_dimension, activation=tup[2],  kernel_initializer=tup[1]))
    for i in range(1, len(layers)):
        tup = layers[i]
        model.add(Dense( tup[0], activation=tup[2],  kernel_initializer=tup[1]))
    #optimizer =  ks.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.1, nesterov=True)
    optimizer = ks.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # w = model.get_weights()
    # w[0][0,1] = 10
    # model.set_weights(w)
    # outputTensor = model.output
    # listOfVariableTensors = model.trainable_weights
    # gradients = k.gradients(outputTensor, listOfVariableTensors)
    # print("gradients", gradients)
    # trainingExample = np.random.random((1, 8))
    # sess = tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())
    #

    #print("INPUT", model.input)

    #evaluated_gradients = sess.run(gradients, feed_dict={model.input: })
    #print("EVALUATED GRADIENTS",evaluated_gradients)
    #sess.close()


    #print("weigth",model.get_weights())
    return model, optimizer

def cplexmodel(jobs, machines, sequences):
    cplexmodel = Model(name = 'simple_job_shop')

    t = []

    for i in range(jobs):
        name = "t" + str(i)+"_"
        tj = cplexmodel.continuous_var_list(
            range(machines), lb = 0, name = name)
        t.append(tj)

    z = []
    for j in range(jobs):
        zj = []
        for k in range(jobs):
            name = "z"+str(j)+"_"+str(k)
            zjk = cplexmodel.binary_var_list(
                range(machines), name = name
            )
            zj.append(zjk)
        z.append(zj)
    m = cplexmodel.continuous_var(name = 'makespan')

    bigM = 0
    for seq in sequences:
        for s in seq:
            bigM = bigM + s[1]
    bigM = bigM*2
    #####CONSTRAINTS####
    for j in range(jobs):
        tj = t[j]
        seq = sequences[j]
        for i in range(1, len(seq)):
            ohj = seq[i][0]
            ohm1j = seq[i-1][0]
            pohm1j = seq[i-1][1]
            xohj = tj[ohj]
            xohm1j = tj[ohm1j]
            con = cplexmodel.add_constraint( xohj >= xohm1j + pohm1j)
            #print("CIAO",cplexmodel.pprint_as_string(con))
            if i == len(seq) -1:
                pohj = seq[i][1]
                cplexmodel.add_constraint( m >= xohj + pohj)

    for j in range(jobs):
        seqj = sequences[j]
        for k in range(jobs):
            if j != k:
                seqk = sequences[k]
                mach = machmerge(seqj, seqk)
                for machine in mach:
                    xij = t[j][machine]
                    #print(xij)
                    xik = t[k][machine]
                    #print(xik)
                    pij = 0
                    pik = 0
                    for s in seqj:
                        if s[0] == machine:
                            pij = s[1]
                            break
                    for s in seqk:
                        if s[0] == machine:
                            pik = s[1]
                            break

                    cplexmodel.add_constraint(xij >= xik + pik - bigM * z[j][k][machine])
                    cplexmodel.add_constraint(xik >= xij + pij - bigM + bigM * z[j][k][machine])

    #####Objective
    cplexmodel.minimize( m )
    #####Solution
    start = time()
    #print(cplexmodel.export_as_lp_string())

    solution = cplexmodel.solve(log_output = True)

    return (time() - start), solution, cplexmodel.objective_value

def machmerge( seqj, seqk):
    lista = []
    listb = []
    ##lista.append( seqj[i][0] for i in range(len(seqj)))
    for s in seqj:
        lista.append(s[0])
    lista = np.array(lista)
    #listb.append( seqk[i][0] for i in range(len(seqk)))
    for s in seqk:
        listb.append(s[0])
    listb = np.array(listb)
    intersection = np.intersect1d(lista, listb)
    #print("lista", lista,"\nlistb",listb,"\ninter", intersection)
    return intersection

def mainbalerion():
    seed = 300#70
    np.random.seed(seed)

    ############################################create instances###
    jobs = 2
    machines = 2
    sequences, n = createInstances(jobs = jobs, machines = machines)

    state_dimension = n + machines
    instance_dimension = n + machines + n + machines + machines
                     #jobs + machines + nextjobs + nextmachines + action
    print("State Dimension: ", state_dimension, "  Instance dimension: ", instance_dimension)

    ###############################################create model###
    innerlayers = [
        (instance_dimension*10+machines, 'uniform', 'relu'),
        #(2*state_dimension+machines, 'uniform', 'relu'),
        #(instance_dimension, 'uniform', 'relu'),
        (machines, 'uniform', 'sigmoid')
    ]

    outerlayers = [
        (instance_dimension+machines, 'uniform', 'relu'),
        (2*state_dimension, 'uniform', 'relu'),
        # (instance_dimension, 'uniform', 'relu'),
        (1, 'uniform', 'linear')
    ]
    #model, optimizer = createModel(  layers, instance_dimension ) #fi you need to check particular settings
                                                                 # for the optimizer please go deeper into the method

    #####################################Reinforcement Learning###

    min_experience = 1
    max_experience =  2
    min_gamma = 4
    max_gamma = 5

    M = 30

    record = []


    for experience in range(min_experience, max_experience):
        for gamma in range(min_gamma, max_gamma):


            seed = gamma*experience

            outermodel, outeroptimizer = createModel(outerlayers, instance_dimension+machines)  # fi you need to check particular settings

            innermodel, inneroptimizer = createModel(innerlayers, state_dimension+machines)  # fi you need to check particular settings

            # for the optimizer please go deeper into the method
            gamma = gamma/10
            trainer = balerion(
                innermodel,
                inneroptimizer,
                outermodel,
                outeroptimizer,
                jobs,
                machines,
                sequences,
                path=None,
                M=M,
                T=jobs * machines,
                peps=0.9,
                delta=0.75,
                gamma=gamma,
                experience=experience,
                size=10000000000,
                randomized=False,
                finalreward=200*2,#1e1,
                batch_size=5,
                verbose=True,
                npseed=seed)
            episode, reward, rewards, solution, elapsedtime, timings = trainer.solve()
            makespan = 0
            for timing in timings:
                makespan = makespan + timing
            print("Experience", experience," Gamma",gamma, " Best reward", reward," MAKESPAN", makespan, " Elapsed time", elapsedtime)

            record.append( ( experience, gamma, episode, reward, rewards, solution, elapsedtime, timings ) )
                            #   0          1       2       3        4         5          6          7

    ###show resuts###

    fig = plt.figure(1, figsize = (10,4.8))
    chart = fig.add_subplot(111)
    for i in record:
        name = str(i[0]) + "-"+ str(i[1])
        #y = gaussian_filter1d(i[4], sigma=1)
        x = np.array(range(len(i[4])))
        x_smooth = np.linspace(x.min(), x.max(), M)
        y_smooth = spline(x, np.array(i[4]), x_smooth)
        chart.plot( x_smooth, y_smooth, label = name)
    #chart.legend()
    plt.show()

    # episode, reward, rewards, solution, elapsedtime, timings = trainer.play()
    # makespan = 0
    # for timing in timings:
    #     makespan = makespan + timing
    # print("PLAY Experience", experience, " Gamma", gamma, " Best reward", reward, " MAKESPAN", makespan,
    #       " Elapsed time",
    #       elapsedtime)

    elaptime, sol, obj =cplexmodel(jobs, machines, sequences)

    print("CPLEX_SOL  Elapsed time", elaptime, " MAKESPAN", obj)





    return


mainbalerion()
