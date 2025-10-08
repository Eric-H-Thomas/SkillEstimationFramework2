import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
import code

def draw_board(ax):
    #Draw the bullseye rings and scoring rings
    radii = [6.35, 15.9, 99, 107, 162, 170]
    for r in radii: 
        circle = Circle((0,0),r,fill=False)
        ax.add_artist(circle)

    #Do the radii 
    start_d = 15.9
    end_d = 170.0
    angle_increment = math.pi / 10.0
    angle = -angle_increment / 2.0

    for i in range(20):
        sx = start_d * math.cos(angle)
        sy = start_d * math.sin(angle)
        dx = end_d * math.cos(angle)
        dy = end_d * math.sin(angle)
        plt.plot([sx, dx], [sy, dy], color="Black")
        # print 'Angle = ', 180.0*angle/math.pi
        angle += angle_increment

def label_regions():
    slices = [11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]

    angle_increment = math.pi / 10.0    
    angle = math.pi
    r = 130.0

    for i in range(20):
        x = r*math.cos(angle)
        y = r*math.sin(angle)
        plt.text(x,y,str(slices[i]),fontsize=12,horizontalalignment='center')
        angle += angle_increment

def score(x,y):
    """ 
    Return the score for location (x,y) on the dartboard. 
    Units are mm    
    """
    
    #First convert to polar coordinates to get distance from (0,0) and angle from (0,0)
    a = math.atan2(y,x) #angle
    r = math.hypot(x,y) #radius

    scaling = 1.0

    double = False

    #Figure out which distance we fall in 
    if r < 6.35:
        #Double bullseye = 50 points
        return 50.0, double
    if r < 15.9: 
        #Single bullseye = 25 points
        return 25.0, double

    if r > 99 and r < 107:
        # Triple score
        scaling = 3.0

    if r > 162 and r < 170:
        # Double score
        double = True
        scaling = 2.0

    if r > 170:
        # Off the board
        return 0.0, double

    #Figure out which slice we fall in
    angle_increment = math.pi / 10.0
    slice_low = - math.pi - angle_increment / 2.0
    slice_high = slice_low + angle_increment

    slices = [11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]

    for i in range(21):
        if a > slice_low and a < slice_high:
            return scaling*slices[i], double
        slice_low += angle_increment
        slice_high += angle_increment

    #Check for 11 slice

    #Must have missed the board!
    return 0.0, double


def npscore(x,y,return_doub=False):

    slices = np.array([11,8,8,16,16,7,7,19,19,3,3,17,17,2,2,15,15,10,10,6,6,13,13,4,4,18,18,1,1,20,20,5,5,12,12,9,9,14,14,11,11])

    a = np.arctan2(y,x)
    r = np.hypot(x,y)

    a = ((a + math.pi) * (40/(2*math.pi))).astype(int)
    ans = slices[a]
    
    doubb = r<6.35
    bulls = r<15.9
    on = r<170
    
    trip = (r>99) & (r<107)
    doub = (r<170) & (r>162)

    ans *= np.invert(bulls)*on 
    ans += (25*bulls) + (25*doubb)
    ans += (ans*trip*2) + (ans*doub)
    
    if return_doub:
        return ans, doub+doubb
    return ans 

def getSampleEV(a,value,startScore,N,xSkill):
    ev = 0.0
    aimScore, aimD = score(a[0], a[1])
    print("Getting sample EV for ", a, 'aiming at', aimScore, aimD)
    scoreSet = dict()
    for n in range(N):
        xa,ya = np.random.multivariate_normal(a, xSkill*np.identity(2))
        curScore, double = score(xa, ya)
        key = str(curScore) + str(double)[0]
        if key not in scoreSet:
            scoreSet[key] = 0
        scoreSet[key] += 1
        nextScore = startScore - curScore
        if nextScore < 0 or nextScore == 1:
            nextScore = startScore 
        if nextScore == 0:
            if not double:
                nextScore = startScore 
        ev += value[int(nextScore)]

    print("Sample score set: ")
    sortSet = sorted(scoreSet.items(), key=lambda x: x[1], reverse=True)
    i = 0
    for x in sortSet:
        print(" ", x[0], ':', x[1]/N)
        i += 1
        if i > 10:
            break
    return ev / N

def get_values(resolution, curScore, values):
    X = np.arange(-170.0, 171.0, resolution)
    Y = np.arange(-170.0, 171.0, resolution)
    V = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            s, double = score(X[i],Y[j])
            newScore = curScore - s
            #Did we bust (score too much)?
            # Less than 0 or exactly 1
            if newScore < 0 or newScore == 1:
                newScore = curScore
            #Did we double out correctly?
            if newScore == 0:
                if not double:
                    newScore = curScore
            V[i,j] = values[int(newScore)]
    return X,Y,V

def get_symmetric_normal_distribution(cov, resolution):
    #Determine what 4 standard deviations is (1/15,787 will fall outside this)
    std = np.ceil(np.sqrt(cov))
    boundary = 4*std
    #print("Covariance = ", cov, 'std = ', std, 'boundary = ', boundary)
    # X = np.arange(-boundary, boundary+1, resolution)
    # Y = np.arange(-boundary, boundary+1, resolution)
    
    X = np.arange(-170.0,171.0,resolution)
    Y = np.arange(-170.0,171.0,resolution)


    D = np.zeros((len(X), len(Y)))
    N = multivariate_normal([0.0,0.0],cov=cov)
    # print(N.cov)
    for i in range(len(X)):
        for j in range(len(Y)):
            D[i,j] = N.pdf([X[i],Y[j]])
    D *= resolution**2
    # code.interact("...", local=dict(globals(), **locals()))
    return X,Y,D

#Get all of the possible actions in our domain
def getActions():
    actions = [(0,0)]

    #Distances are center, mid-single, triple, mid-single, double
    distances = [62, 103, 133, 166]
    angle_increment = math.pi / 10.0
    angle = 0

    for wedge in range(20):
        for d in distances:
            #Get (x,y) location for this target)
            x = d*math.cos(angle)
            y = d*math.sin(angle)
            actions.append((x,y))

    return actions

def plotValueFunction(value):
    plt.scatter(list(range(len(value))), value)
    plt.show()

def valueIteration(start_score, xskill, valueFile):
    #xskill is the variance of their noise
    #Initialize the value function
    value = [-100.0]*(start_score+1)
    value[0] = 0.0 #Value of state 0 (we are done!)
    value[1] = 0.0 #value of state 1 (never get here!)
    delta = 10.0 #How much did it change this iter?
    gamma = 1.0 #Discount factor
    tolerance = 0.01 #When do we stop?
    resolution = 5.0
    iters = 0

    while delta > tolerance:
        delta = 0.0 #reset delta

        #print('Starting iteration', iters)
        #print('Displaying current value function')
        #plotValueFunction(value)
        #print('  Value of score = 201: ', value[-1])

        for s in range(2,len(value)):
            #Update the value of this state
            #First, what is the EV of the best action?
            #print("Updating score for state", s)
            Xn,Yn,Zn = get_symmetric_normal_distribution(xskill,resolution)
            Xs,Ys,Zs = get_values(resolution, s, value)

            #Convolve to produce the EV and aiming spot
            EV = convolve2d(Zs,Zn,mode="same",fillvalue=value[s])
            # print("Convolving with fill value = ", value[s])

            # print('Shape of Distribution: ', Zn.shape)
            # print('Shape of Values: ', Zs.shape)
            # print('Shape of EV: ', EV.shape)

            # code.interact("vi()...", local=dict(globals(), **locals()))


            #Get maximum of EV
            mxi, myi = np.unravel_index(EV.argmax(), EV.shape)
            
            #Best aiming point
            mx = Xn[mxi]
            my = Yn[myi]

            #Plot out the EV convolution result
            # plt.imshow(EV)
            # plt.show()


            #Print out the best location and value.  Compare with sampling
            # N = 1000
            # print("Max point: ", mx, my)
            # aimScore, aimD = score(mx,my)
            # print("  Score of max-point: ", aimScore)
            # print("  Value of hitting that target: ", value[int(s-aimScore)])
            # print("Max EV: ", EV.max())
            # sampleEV = getSampleEV((mx,my),value,s,N,xskill)
            # print("Sample EV with N = ", N, "is: ", sampleEV)
            # print("Ratio between max and sample: ", EV.max()/sampleEV)


            #How much are we going to change the value?
            current_delta = abs(value[s] + 1 - gamma*EV.max()) 
            if current_delta > delta:
                delta = current_delta

            #Update the value
            print('  [', s, '] : old = ', value[s], ' max EV = ', EV.max())
            value[s] = -1 + gamma*EV.max()
            print('            : new = ', value[s])

            # plt.figure()
            # ax = plt.gca()
            # draw_board(ax)
            # label_regions()    
            # plt.axis('equal')
            # plt.scatter([mx],[my])
            # plt.show()

        print('   delta = ', delta)
        iters += 1

    print('Value Iteration Converged after', iters, 'iterations')
    print('Final Value Function')
    np.savetxt(valueFile, value, delimiter=',')
    plotValueFunction(value)
    return value

def validateValueFunction(value, xSkill, startScore, N, resolution):
    counts = [0.0]*(startScore+1)
    totalTurns = [0.0]*(startScore+1)

    fullGameThrows = 0

    #Run N games
    for n in range(N):
        throws = 0
        S = startScore
        scoreSequence = [startScore]
        while S > 0:
            Xn,Yn,Zn = get_symmetric_normal_distribution(xskill,resolution)
            Xs,Ys,Zs = get_values(resolution, S, value)

            #Convolve to produce the EV and aiming spot
            EV = convolve2d(Zs,Zn,mode="same",fillvalue=value[int(S)])

            #Get maximum of EV
            mxi, myi = np.unravel_index(EV.argmax(), EV.shape)
            #Best aiming point
            mx = Xn[mxi]
            my = Yn[myi]

            #Now throw at that aiming point
            xa,ya = np.random.multivariate_normal((mx,my), xSkill*np.identity(2))
            
            # curScore, double = score(xa, ya)
            curScore,double = npscore(xa,ya,return_doub=True)

            nextScore = S - curScore
            if nextScore < 0 or nextScore == 1:
                nextScore = S 
            if nextScore == 0:
                if not double:
                    nextScore = S
            S = nextScore
            scoreSequence.append(S)
            throws += 1 

        print('Done with game', n, ' : (', throws, ') - ', scoreSequence)
        fullGameThrows += throws
        #Figure out the scores for all states
        T = 1.0
        for i in range(len(scoreSequence)-2,-1,-1):
            counts[int(scoreSequence[i])] += 1.0 
            totalTurns[int(scoreSequence[i])] += T
            T += 1.0

    print('Done with validation')
    print(' Value of state ', len(value)-1, '=', value[-1])
    print(' Average number of turns =', fullGameThrows / N)
    print('FULL VALUE COMPARISON: ')
    for s in range(startScore):
        if counts[s] != 0:
            print('   V[', s, '] = ', value[s])
            print('   MC-V[', s, '] = ', totalTurns[s]/counts[s])
            print('      Counts = ', counts[s], ' | TT = ', totalTurns[s])



if __name__ == "__main__":
    start_score = 201
    xskill = 2.5 #150.5 #This is standard-deviation.  Square before passing in
    valueFile = 'valuefunction-' + str(xskill**2) + 'csv'

    try: 
        value = np.loadtxt(valueFile, delimiter=',')
    except: 
        print("No value function stored for skill level (std) of ", xskill, '. Computing one.')
        value = valueIteration(start_score, xskill**2, valueFile)        

    print('Validating value function')
    validateValueFunction(value, xskill**2, start_score, 1000, resolution = 5.0)
    code.interact("...", local=dict(globals(), **locals()))


