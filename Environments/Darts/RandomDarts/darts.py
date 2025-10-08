import numpy as np
import scipy
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as mcolors
import matplotlib
import itertools
import sys
import time 
import os
import code
from matplotlib.cm import ScalarMappable

#This Defines the 1D Darts environment. 
#It also has utilities to help agents in the domain
m = 10
 
def getDomainName():
    return "1d"

def getNoiseModel(rng,L):

    N = rng.normal(0.0,L)
    #print N.cov

    return N

def plot_state_allInfo(states,gameInfo,results_folder):

    # gameInfo -> agent | intended actions | expected_rewards | noisy actions | rewards | resampled_rewards

    # colors = ["r", "g","b","c","m","y"]
    # colors = cm.rainbow(np.linspace(0, 1, len(gameInfo)))

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = colors.keys()
    
    for each in range(len(states)):

        fig = plt.figure()
        ax = plt.subplot(111)

        plot_state(states[each])

        for a in range(len(gameInfo)): 

            aName = gameInfo[a][0].split("-X")[0].split("Agent")[0]
            aNameSplit = gameInfo[a][0].split("-")

            # print aNameSplit

            noise = aNameSplit[1].replace("X","")
            # print noise


            plt.plot(gameInfo[a][1][each], gameInfo[a][2][each], label =  aName + "-" + aNameSplit[-1], linestyle = 'None', marker = "s", color = colors[a])
            plt.plot(gameInfo[a][3][each], gameInfo[a][4][each], linestyle = 'None', marker = "P", color = colors[a])
            plt.plot(gameInfo[a][3][each], np.mean(gameInfo[a][5][each]), linestyle = 'None', marker = "*", color = colors[a])

        plot_ev(states[each],float(noise),color = "k")

        # Shrink current axis by 10%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':14})


        #plt.show()
        plt.title("xSkill: " + str(aNameSplit[-2]))
        plt.savefig(results_folder + os.path.sep + aNameSplit[-2] + "-state" + str(each) + ".png", bbox_inches='tight')


        plt.clf()
        plt.close(fig)

def plot_state(S,color="black"):
    #Plot the underlying state
    #Get X and Y points
    low = True

    X = [-m]
    Y = [0.0]

    for s in S:
        X.extend([s,s])
        if low:
            Y.extend([0.0,1.0])
            low = False
        else:
            Y.extend([1.0,0.0])
            low = True

    X.extend([S[-1],m])
    Y.extend([0.0,0.0])
    plt.plot(X,Y,label = "Reward", color = color)

def getRewardsForPlot(S):
        
    evs = []

    #Plot the underlying state
    #Get X and Y points
    low = True

    X = [-m]
    Y = [0.0]

    for s in S:
        X.extend([s,s])
        if low:
            Y.extend([0.0,1.0])
            low = False
            evs.append(1)
        else:
            Y.extend([1.0,0.0])
            low = True
            evs.append(0)

    X.extend([S[-1],m])
    Y.extend([0.0,0.0])
    #plt.plot(X,Y,label = "Reward", color = color)

    return evs

def plot_ev(S,L,color='r',label=None):
    #Plot the actual expected value for each action in the space
    #using noise level L and NS number of monte-carlo samples for estimating the value
    EV, A = convolve_ev(S,L)

    if label is not None:
        plt.plot(A,EV,color,label=label)
    else:
        plt.plot(A,EV,color)

def get_v(rng,S,a):
    ''' Returns the value of a given state ''' 
    
    low = True
    for s in S:
        if a < s:
            break
        low = not low

    if low:
        return 0.0
    else:
        return 1.0

def findRegion(S,a):
    ''' Returns the index of region (state interval) that contains the given action ''' 
    
    for each in range(len(S)):
        # except last element to avoid error
        if each != len(S)-1:
        # 0 & 1 | 1 & 2 and so on except last
            if a >= float(S[each]) and a <= float(S[each + 1]):
                return each, each + 1

    # edge cases/regions
    if a >= -m and a <= float(S[0]):
        return -m, 0

    elif a >= float(S[len(S)-1]) and a <= m:
        return len(S)-1, m
    
def checkIfActionInRegion(S,a,l,r):

    # somewhere in the middle
    if a >= float(S[l]) and a <= float(S[r]):
        return True

    else:
        # first edge
        if l == -m:
            if a >= -m and a <= float(S[r]):
                return True
            else:
                return False
        # last edge
        elif r == m:
            if a >= float(S[l]) and a <= m:
                return True
            else: return False
        else:
            return False

def get_rand_reward(S):
    
    lenR = 0

    # print "State: ", S

    # for each one of the successful regions on the given state - (incremented by 2)
    for i in range(0,len(S),2):

        # compute the len of the region and accumulate
        lenR += abs(S[i] - S[i+1])

    # Compute mean of rewards
    rand_reward = lenR / 20.0

    return rand_reward

def wrap_action(a):
    while a > m:
        a = a - 2*m
    while a < -m:
        a = a + 2*m
    return a

def sample_action(rng,S,L,a,noiseModel=None):
    # Noisy action
    # print 'Sampling 1 for ', a, L

    # If noise model was not given, proceed to get it
    if noiseModel == None:
        noise = getNoiseModel(rng,L)
    # Otherwise, use given noise model
    else:
        noise = noiseModel

    na = a + noise

    na = wrap_action(na)

    return na

def actionDiff(a1,a2):
    d = a1 - a2
    if d > m:
        d -= 2*m
    if d < -m:
        d += 2*m

    return d

def sample_1(rng,S,L,a):
    # See where noisy action lands in S
    return get_v(rng,S,sample_action(rng,S,L,a))

def sample_N(rng,S,L,NS,a):
    # print 'Sampling N for', a, L
    tr = 0.0
    for i in range(NS):
        tr += sample_1(rng,S,L,a)
    return tr / float(NS)

def convolve_ev(rng,S,L,delta=1e-2):
    # Get representation of function
    num_points = int(6*m/delta)
    big_grid = np.linspace(-3*m,3*m,num_points)

    state = [get_v(rng,S,wrap_action(a)) for a in big_grid]

    # Get convolver
    err = stats.norm(loc=0,scale=L)
    errpmf = err.pdf(big_grid)*delta

    conv_pmf = np.convolve(state,errpmf,'same')
    
    left = int(num_points/3)
    right = int(2*left)

    return conv_pmf[left:right], big_grid[left:right]

def get_N_states(rng,low,high,N,min_width=0.0):
    states = []

    for n in range(N):
        #Create N regions (where N is even)
        num_r = rng.integers(low,high)*2

        rs = 0
        S = []

        while rs < num_r:
            #Get the N boundary points
            ns = rng.uniform(-m, m)
            valid_point = True
            for s in S:
                if abs(ns-s) < min_width:
                    valid_point = False
                    break
            if valid_point:
                S.append(ns)
                rs += 1

        S = np.sort(S)
        #Filter out points too close together
        states.append(S.tolist())

    return states

def get_target(rng,S,L,delta): 
    ''' Get the target for a given state and xskill level '''   

    # Do convolution with resolution of "delta"
    va, a = convolve_ev(rng,S,L,delta)

    # Get the index of the target with the given (max) value
    i = np.argmax(va)
    
    # return target that will give max ev and the actual ev
    return a[i], va[i]

def get_all_targets(rng,S,L,delta):

    # Do convolution with resolution of "delta"
    va, a = convolve_ev(rng,S,L,delta)

    # Get the index of the target with the given (max) value
    i = np.argmax(va)

    # return target that will give max ev and the actual ev
    # as well as all the other targets and evs (all the information from the convolution)
    # return a,va,a[i],va[i]
    return va,a[i],va[i]

def verifyConvolveEV(xskills,state):

    for j in range(len(xskills)):
        EVs, A = convolve_ev(state,xskills[j])
        
        indexMax = np.argmax(EVs)
        a = A[indexMax]

        print(f"xskill: {xskills[j]}")
        print(f"action: {a}")

        aSum = 0.0
        n = 10000
        for n in range(n):
            na = sample_action(rng,state,xskills[j],a)
            v = get_v(state,na)
            # print(f"\tna #{n+1}: {na} | values: {v}")

            aSum += v


        avg = aSum/n
        print(f"avg: {avg} | EV: {EVs[indexMax]}\n")

    # code.interact("...", local=dict(globals(), **locals()))

def testHits(xskills,state,numTries,aim=""):

    allPercentHits = []

    print(f"Aiming at: {aim}")

    for xs in xskills:

        EVs, A = convolve_ev(state,xs)
                
        if aim == "optimal":
            indexMax = np.argmax(EVs)
            a = A[indexMax]
        else:
            a = 0.0

        # print(f"xskill: {xs}")
        # print(f"action: {a}")
        
        hits = 0.0

        for tries in range(int(numTries)):

            na = sample_action(rng,state,xs,a)
            v = get_v(state,na)

    
            # Verify if the action hits the board or not
            if not (na < -m or na > m):
                hits += 1.0

        percentHit = (hits/numTries)*100.0
        allPercentHits.append(percentHit)
        
        print("xSkill: ", xs, "| \tTotal Hits: ", hits, " out of ", numTries, "-> ", percentHit, "%")


    '''
    plt.plot(xskills,allPercentHits)
    plt.xlabel('xSkills')
    plt.ylabel('% Hits')
    #plt.legend()
    plt.show()
    '''

    return allPercentHits


if __name__ == '__main__':

    ##################################################
    # PARAMETERS FOR PLOTS
    ##################################################

    # plt.rcParams.update({'font.size': 14})
    # plt.rcParams.update({'legend.fontsize': 14})
    plt.rcParams["axes.titleweight"] = "bold"

    #plt.rcParams["font.weight"] = "bold"
    #plt.rcParams["axes.labelweight"] = "bold"

    ##################################################


    # seed = np.random.randint(0,1000000,1)
    seed = 10
    rng = np.random.default_rng(seed)

    folder = f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
    
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(folder+"Plots-States"):
        os.mkdir(folder+"Plots-States")

    if not os.path.exists(folder+f"Plots-States{os.sep}Wrap"):
        os.mkdir(folder+f"Plots-States{os.sep}Wrap")


    # xskills = np.round(np.linspace(0.5,4.5,num=5),4)
    xskills = np.round(np.linspace(0.5,20.0,num=10),4)

    colors = ["tab:orange","tab:green","tab:red","tab:purple","tab:pink"]
    colors += ["tab:brown","tab:gray","tab:olive","tab:cyan","b"]
    # colors = cm.rainbow(np.linspace(0,1,len(xskills)))

    numStates = 5

    states = get_N_states(rng,3,5,numStates,0.5)


    numTries = 10_000
    
    for i in range(1): #(numStates):
        # verifyConvolveEV(xskills,states[i])

        allPercentHits = testHits(xskills,states[i],numTries,aim="optimal")
        allPercentHits = testHits(xskills,states[i],numTries,aim="middle")

    
    code.interact("...", local=dict(globals(), **locals()))


    for j in range(len(xskills)):
        EVs, A = convolve_ev(states[0],xskills[j])
        
        indexMax = np.argmax(EVs)
        a = A[indexMax]

        print(f"xskill: {xskills[j]}")
        print(f"action: {a}")

        aSum = 0.0
        n = 10000
        for n in range(n):
            na = sample_action(rng,states[0],xskills[j],a)
            v = get_v(states[0],na)
            # print(f"\tna #{n+1}: {na} | values: {v}")

            aSum += v


        avg = aSum/n
        print(f"avg: {avg} | EV: {EVs[indexMax]}")

        code.interact("...", local=dict(globals(), **locals()))



    for i in range(numStates):

        # '''
        fig = plt.figure()
        ax = plt.gca()
        #ax = plt.subplot2grid((5,2), (0, 0))

        plot_state(states[i], color = "tab:blue")

        plt.margins(0.10) 
        ax.autoscale(True)

        plt.ylim(-0.1,1.1)

        for j in range(len(xskills)):
            plot_ev(states[i],xskills[j],color=colors[j],label=xskills[j])


        plt.legend()

        plt.tight_layout()
        plt.savefig(folder+f"Plots-States{os.sep}Wrap{os.sep}1D-state" + str(i)+f"-EVs-WrapTrue.png")
       
        plt.close()
        plt.clf()
        # '''


        # '''
        for j in range(len(xskills)):

            if not os.path.exists(f"{folder}Plots-States{os.sep}Wrap{os.sep}xskill{xskills[j]}{os.sep}"):
                os.mkdir(f"{folder}Plots-States{os.sep}Wrap{os.sep}xskill{xskills[j]}{os.sep}")

            fig = plt.figure()
            ax = plt.gca()

            EVs, A = convolve_ev(states[i],xskills[j])

            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(min(EVs),max(EVs))
            sm = ScalarMappable(norm = norm, cmap = cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm,ax=ax)
            cbar.ax.set_title("EVs")

            for ii in range(len(A)):
                plt.vlines(x = A[ii], ymin = 0, ymax = EVs[ii],colors = cmap(norm(EVs[ii])))
            
            plot_state(states[i], color = "tab:blue")
            plt.title(f"State: {i} | Xskill: {xskills[j]}")
            plt.tight_layout()
            plt.savefig(f"{folder}Plots-States{os.sep}Wrap{os.sep}xskill{xskills[j]}{os.sep}1D-state{i}.png")
            
            plt.close()
            plt.clf()

        # '''


        '''
        from plotly import graph_objs as go
        import plotly as py

        rewards = getRewardsForPlot(states[i])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=states[i], y=rewards, mode='markers', marker_size=20,
        ))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, 
                         zeroline=True, zerolinecolor='black', zerolinewidth=3,
                         showticklabels=False)
        #fig.update_layout(height=200, plot_bgcolor='white')
        fig.layout.update(height=200, plot_bgcolor='white')

        # Save plotly
        unique_url = py.offline.plot(fig, filename= "1D-state" + str(i)+".html", auto_open=False)

        '''

        '''
        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(8, 3))
        ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])



        rewards = getRewardsForPlot(states[i])
        # print(rewards)

        colors = ["w"]
        for r in rewards:
            if r == 0:
                colors.append("w")
            else:
                colors.append("tab:blue")

        # The second example illustrates the use of a ListedColormap, a
        # BoundaryNorm, and extended ends to show the "over" and "under"
        # value colors.
        cmap = matplotlib.colors.ListedColormap(colors)
        # cmap.set_over('0.25')
        # cmap.set_under('0.75')

        # If a ListedColormap is used, the length of the bounds array must be
        # one greater than the length of the color list.  The bounds must be
        # monotonically increasing.
        bounds = [-10] + states[i] + [10]
        print(bounds)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                        norm=norm,
                                        # to use 'extend', you must
                                        # specify two extra boundaries:
                                        boundaries= bounds ,
                                        # extend='both',
                                        ticks=bounds,  # optional
                                        spacing='proportional',
                                        orientation='horizontal')
        # cb2.set_label('Discrete intervals, some other units')

        # clb = plt.colorbar()
        # cb2.set_label('1', labelpad=-82, y=-500.0, rotation=0)

        plt.text(-0.02,1,"1")
        plt.text(-0.02,0,"0")

        degrees = 45 #90
        plt.xticks(rotation=degrees)


        #plt.show()
        plt.savefig("1D-state" + str(i)+".png")
        '''
