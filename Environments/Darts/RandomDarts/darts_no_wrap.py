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
import json
import os
import code
from matplotlib.cm import ScalarMappable


#This Defines the 1D Darts environment.
# WITH NO WRAP FUNCTIONALITY
#It also has utilities to help agents in the domain

m = 10
 
def get_domain_name():
    return "1d"

def draw_noise_sample(rng,L):

    N = rng.normal(0.0,L)
    #print N.cov

    return N

def plot_reward_profile(rng,S,color="black"):
    #Plot the underlying state
    #Get X and Y points
    low = True

    X = [-m,-m]
    Y = [0.0,1.0]

    for i in range(len(S)):
        s = S[i]

        if i < len(S)-1:
            a = (S[i]+S[i+1])/2
            X.extend([s,s,S[i+1]])
        else:
            a = (S[i]+m)/2
            X.extend([s,s,s])

        v = get_reward_for_action(rng,S,a)

        # print(s)
        # print("a: ",a," | v:",v)


        if low:
            Y.extend([1.0,v,v])
            low = False
        else:
            Y.extend([v,v,1.0])
            low = True

    X.extend([S[-1],m])
    Y.extend([1.0,1.0])

    X.extend([m,m])
    Y.extend([1.0,0.0])

    plt.plot(X,Y,label = "Reward", color = color)
    # plt.scatter(X,Y,label = "Reward", color = "black")

'''
# Need to update
def get_rewards_for_plot(S):
        
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
'''

def plot_expected_values(rng,S,L,color='r',label=None):
    #Plot the actual expected value for each action in the space
    #using noise level L and NS number of monte-carlo samples for estimating the value
    EV, A = compute_expected_value_curve(rng,S,L)

    if label is not None:
        plt.plot(A,EV,color,label=label)
    else:
        plt.plot(A,EV,color)

def get_reward_for_action(rng,S,a):
    # Returns the value of a given action on a given state 
    
    # Verify first if action outside of the range 
    # (Not falling withhin any of the regions)
    if a > m or a < -m:
        return 0.0

    low = True
    for w in range(len(S)):
        s = S[w]

        if a < s:
            break
        low = not low

    if low:
        return 1.0 #0.0
    else:

        '''
        if w < len(S)-1:
            # num = rng.integers(s,S[i+1])
            num = abs((s+S[w+1])/2) * len(S)
        else:
            # num = rng.integers(s,m)
            num = abs((s+m)/2) * len(S)
        
        # (num - lower_bound)%(upper_bound-lower_bound) + lower_bound
        lowerBound = 2
        upperBound = 6
        v = int((num-lowerBound) % (upperBound-lowerBound) + lowerBound)*1.0
        # print(v)# 
        '''  

        v = 2.0

        #code.interact("...", local=dict(globals(), **locals()))

        return v

def sample_noisy_action(rng,S,L,a,noiseModel=None):
    # Noisy action

    # If noise model was not given, proceed to get it
    if noiseModel == None:
        noise = draw_noise_sample(rng,L)
    # Otherwise, use given noise model
    else:
        noise = noiseModel

    na = a + noise

    return na

def calculate_action_difference(a1,a2):
    d = a1 - a2

    return d

def sample_single_rollout(rng,S,L,a):
    # See where noisy action lands in S
    return get_reward_for_action(rng,S,sample_noisy_action(rng,S,L,a))

def estimate_value_with_samples(rng,S,L,NS,a):
    # print 'Sampling N for', a, L
    tr = 0.0
    for i in range(NS):
        tr += sample_single_rollout(rng,S,L,a)
    return tr / float(NS)

def compute_expected_value_curve(rng,S,L,delta=1e-2):
    # Get representation of function
    num_points = int(6*m/delta)
    big_grid = np.linspace(-3*m,3*m,num_points)

    state = [get_reward_for_action(rng,S,a) for a in big_grid]

    # Get convolver
    err = stats.norm(loc=0,scale=L)
    errpmf = err.pdf(big_grid)*delta

    conv_pmf = np.convolve(state,errpmf,'same')

    left = int(num_points/3)
    right = int(2*left)

    return conv_pmf[left:right], big_grid[left:right]

def generate_random_states(rng,low,high,N,min_width=0.0):
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

def get_optimal_action_and_value(rng,S,L,delta): 
    ''' Get the target for a given state and xskill level '''   

    # Do convolution with resolution of "delta"
    va, a = compute_expected_value_curve(rng,S,L,delta)

    # Get the index of the target with the given (max) value
    i = np.argmax(va)
    
    # return target that will give max ev and the actual ev
    return a[i], va[i]

def get_expected_values_and_optimal_action(rng,S,L,delta):

    # Do convolution with resolution of "delta"
    va, a = compute_expected_value_curve(rng,S,L,delta)

    # Get the index of the target with the given (max) value
    i = np.argmax(va)

    # return target that will give max ev and the actual ev
    # as well as all the other targets and evs (all the information from the convolution)
    return a,va,a[i],va[i]
    
def verify_expected_value_convolution(rng,xskills,state):

    for j in range(len(xskills)):
        EVs, A = compute_expected_value_curve(rng,state,xskills[j])
        
        indexMax = np.argmax(EVs)
        a = A[indexMax]

        print(f"xskill: {xskills[j]}")
        print(f"action: {a}")

        aSum = 0.0
        n = 10000
        for n in range(n):
            na = sample_noisy_action(rng,state,xskills[j],a)
            v = get_reward_for_action(rng,state,na)
            # print(f"\tna #{n+1}: {na} | values: {v}")

            aSum += v


        avg = aSum/n
        print(f"avg: {avg} | EV: {EVs[indexMax]}\n")

    # code.interact("...", local=dict(globals(), **locals()))

def simulate_board_hits(xskills,state,numTries,aim=""):

    allPercentHits = []

    print(f"Aiming at: {aim}")

    for xs in xskills:

        EVs, A = compute_expected_value_curve(rng,state,xs)
                
        if aim == "optimal":
            indexMax = np.argmax(EVs)
            a = A[indexMax]
        else:
            a = 0.0

        # print(f"xskill: {xs}")
        # print(f"action: {a}")
        
        hits = 0.0

        for tries in range(int(numTries)):

            na = sample_noisy_action(rng,state,xs,a)
            v = get_reward_for_action(rng,state,na)

    
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

    if not os.path.exists(folder+f"Plots-States{os.sep}NoWrap"):
        os.mkdir(folder+f"Plots-States{os.sep}NoWrap")


    # xskills = np.round(np.linspace(0.5,4.5,num=5),4)
    xskills = np.round(np.linspace(0.5,15.0,num=10),4)

    colors = ["tab:orange","tab:green","tab:red","tab:purple","tab:pink"]
    colors += ["tab:brown","tab:gray","tab:olive","tab:cyan","b"]
    # colors = cm.rainbow(np.linspace(0,1,len(xskills)))


    numStates = 20

    states = generate_random_states(rng,3,5,numStates,0.5)

    '''
    numTries = 100_000
    
    for i in range(1): #(numStates):
        verify_expected_value_convolution(rng,xskills,states[i])
    code.interact("...", local=dict(globals(), **locals()))


    for i in range(1): #(numStates):
        allPercentHits = simulate_board_hits(xskills,states[i],numTries,aim="optimal")
        allPercentHits = simulate_board_hits(xskills,states[i],numTries,aim="middle")
    
    code.interact("...", local=dict(globals(), **locals()))

    '''
    

    '''
    rewards = {}

    numStates2 = 50000
    states2 = generate_random_states(rng,3,5,numStates2,0.5)
    seenRegions = 0

    # Find distribution of rewards
    for i in range(numStates2):

        for s in range(len(states2[i])):

            seenRegions += 1

            if s < len(states2[i])-1:
                a = (states2[i][s]+states2[i][s+1])/2
            else:
                a = (states2[i][s]+m)/2

            v = get_reward_for_action(rng,states2[i],a)

            if v not in rewards:
                rewards[v] = 0.0

            rewards[v] += 1.0

    print(f"Seen a total of {seenRegions} regions across {numStates2} states")
   
    for ri in rewards:
        print(f"{ri} -> {(rewards[ri]/seenRegions)*100.0}%")

    code.interact("...", local=dict(globals(), **locals()))

    '''


    for i in range(numStates):

        fig = plt.figure(figsize=(4,2))
        ax = plt.gca()
        #ax = plt.subplot2grid((5,2), (0, 0))

        plot_reward_profile(rng,states[i], color = "tab:blue")

        plt.xlim(-15,15)
        plt.ylim(0,2.1)

        # plt.margins(0.50) 
        # ax.autoscale(True)

        # plt.ylim(0.9,5.1)

        # for j in range(len(xskills)):
        #     plot_expected_values(rng,states[i],xskills[j],color=colors[j],label=xskills[j])


        plt.legend()

        plt.tight_layout()
        plt.savefig(folder+f"Plots-States{os.sep}NoWrap{os.sep}1D-state" + str(i)+f"-EVs-WrapFalse.png")
       
        plt.close()
        plt.clf()
        
        # code.interact("...", local=dict(globals(), **locals()))

        '''
        for j in range(len(xskills)):

            if not os.path.exists(f"{folder}Plots-States{os.sep}NoWrap{os.sep}xskill{xskills[j]}{os.sep}"):
                os.mkdir(f"{folder}Plots-States{os.sep}NoWrap{os.sep}xskill{xskills[j]}{os.sep}")

            fig = plt.figure()
            ax = plt.gca()

            EVs, A = compute_expected_value_curve(rng,states[i],xskills[j])

            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(min(EVs),max(EVs))
            sm = ScalarMappable(norm = norm, cmap = cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm,ax=ax)
            cbar.ax.set_title("EVs")

            for ii in range(len(A)):
                plt.vlines(x = A[ii], ymin = 0.0, ymax = EVs[ii],colors = cmap(norm(EVs[ii])))
            
            plot_reward_profile(rng,states[i], color = "tab:blue")
            plt.title(f"State: {i} | Xskill: {xskills[j]}")
            plt.tight_layout()
            plt.savefig(f"{folder}Plots-States{os.sep}NoWrap{os.sep}xskill{xskills[j]}{os.sep}1D-state{i}.png")
            
            plt.close()
            plt.clf()

        '''


        '''
        from plotly import graph_objs as go
        import plotly as py

        rewards = get_rewards_for_plot(states[i])

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



        rewards = get_rewards_for_plot(states[i])
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
