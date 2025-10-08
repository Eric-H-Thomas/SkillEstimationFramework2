import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    X = 5               #What is the variance we will use for this experiment
    MAXN = 100          #How many observations for each run
    NN = range(2,MAXN)  #The time steps we will compute the estimate for
    M = 1000            #How many repeats of the experiment

    T = list(NN)                    #time steps for plotting
    EX = np.zeros(len(NN))          #estimates of std based on just X
    A = np.sqrt(X)*np.ones(len(NN)) #actual skill level for plotting
    EY = np.zeros(len(NN))          #estimates of std based just on Y
    EXY = np.zeros(len(NN))         #estimates of std based on X and Y

    for m in range(M):
        #For each experiment
        ni = 0 #time step index

        #Get our random samples for this experiment
        P = np.random.default_rng().multivariate_normal([0,0], [[X,0],[0,X]],MAXN)

        #For each time step
        for N in NN:

            #Compute the estimates from the data up to this time step
            sX = np.std(P[:N,0])    #Estimate from just X
            sY = np.std(P[:N,1])    #Estimate from just Y
            XY = np.concatenate((P[:N,0],P[:N,1]))  #Concatenate X and Y together
            sXY = np.std(XY)        #Estimate from X and Y

            #Add these estimates to our running total to compute average across all repeats of the experiment
            EX[ni] += sX            
            EY[ni] += sY
            EXY[ni] += sXY

            ni += 1     #Increment counter



    #Compute average across experiment for each time step
    EX /= M
    EY /= M
    EXY /= M


    #Plot everything out
    plt.plot(T,EX, label="x-estimate")
    plt.plot(T,EY, label="y-estimate")  
    plt.plot(T,EXY,label="xy")
    plt.plot(T,A, label="actual")
    plt.legend()
    plt.show()