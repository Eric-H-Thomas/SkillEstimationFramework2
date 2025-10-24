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

#This Defines the 1D Darts environment. 
#It also has utilities to help agents in the domain
m = 10
 
def get_domain_name():
    return "1d"

def draw_noise_sample(rng,L):

    N = rng.normal(0.0,L)
    #print N.cov

    return N

def plot_states_with_agent_details(states,gameInfo,results_folder,wrap):

    # gameInfo -> agent | intended actions | expected_rewards | noisy actions | rewards | resampled_rewards

    # colors = ["r", "g","b","c","m","y"]
    # colors = cm.rainbow(np.linspace(0, 1, len(gameInfo)))

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = colors.keys()
    
    for each in range(len(states)):

        fig = plt.figure()
        ax = plt.subplot(111)

        plot_reward_profile(states[each])

        for a in range(len(gameInfo)): 

            aName = gameInfo[a][0].split("-X")[0].split("Agent")[0]
            aNameSplit = gameInfo[a][0].split("-")

            # print aNameSplit

            noise = aNameSplit[1].replace("X","")
            # print noise


            plt.plot(gameInfo[a][1][each], gameInfo[a][2][each], label =  aName + "-" + aNameSplit[-1], linestyle = 'None', marker = "s", color = colors[a])
            plt.plot(gameInfo[a][3][each], gameInfo[a][4][each], linestyle = 'None', marker = "P", color = colors[a])
            plt.plot(gameInfo[a][3][each], np.mean(gameInfo[a][5][each]), linestyle = 'None', marker = "*", color = colors[a])

        plot_expected_values(states[each],float(noise),color = "k",wrap=wrap)

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

def plot_reward_profile(S,color="black"):
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

def plot_expected_values(S,L,color='r',label=None,wrap=True):
    #Plot the actual expected value for each action in the space
    #using noise level L and NS number of monte-carlo samples for estimating the value
    EV, A = compute_expected_value_curve(S,L,wrap=wrap)

    if label is not None:
        plt.plot(A,EV,color,label=label)
    else:
        plt.plot(A,EV,color)

def get_reward_for_action(S,a,wrap=True):
    ''' Returns the value of a given state ''' 
    
    if not wrap:
        # Verify first if action outside of the range 
        # (Not falling withhin any of the regions)
        if a > m or a < -m:
            return 0.0

    low = True
    for s in S:
        if a < s:
            break
        low = not low

    if low:
        return 0.0
    else:
        return 1.0

def find_state_interval(S,a):
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
    
def is_action_within_interval(S,a,l,r):

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

def calculate_random_reward(S):
    
    lenR = 0

    # print "State: ", S

    # for each one of the successful regions on the given state - (incremented by 2)
    for i in range(0,len(S),2):

        # compute the len of the region and accumulate
        lenR += abs(S[i] - S[i+1])

    # Compute mean of rewards
    rand_reward = lenR / 20.0

    return rand_reward

def wrap_action_within_bounds(a):
    while a > m:
        a = a - 2*m
    while a < -m:
        a = a + 2*m
    return a

def sample_noisy_action(rng,S,L,a,noiseModel=None,wrap=True):
    # Noisy action
    # print 'Sampling 1 for ', a, L

    # If noise model was not given, proceed to get it
    if noiseModel == None:
        noise = draw_noise_sample(rng,L)
    # Otherwise, use given noise model
    else:
        noise = noiseModel

    na = a + noise

    if wrap:
        na = wrap_action_within_bounds(na)

    return na

def calculate_action_difference(a1,a2,wrap=True):
    d = a1 - a2

    if wrap:
        if d > m:
            d -= 2*m
        if d < -m:
            d += 2*m

    return d

def sample_single_rollout(rng,S,L,a,wrap):
    # See where noisy action lands in S
    return get_reward_for_action(S,sample_noisy_action(rng,S,L,a,None,wrap),wrap)

def estimate_value_with_samples(rng,S,L,NS,a,wrap):
    # print 'Sampling N for', a, L
    tr = 0.0
    for i in range(NS):
        tr += sample_single_rollout(rng,S,L,a,wrap)
    return tr / float(NS)

# Current version
def compute_expected_value_curve(S,L,delta=1e-2,wrap=True):
    # Get representation of function
    num_points = int(6*m/delta)
    big_grid = np.linspace(-3*m,3*m,num_points)

    if wrap:
        state = [get_reward_for_action(S,wrap_action_within_bounds(a),wrap) for a in big_grid]
    else:   
        state = [get_reward_for_action(S,a,wrap) for a in big_grid]

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

def get_optimal_action_and_value(S,L,delta,wrap=True): 
    ''' Get the target for a given state and xskill level '''   

    # Do convolution with resolution of "delta"
    va, a = compute_expected_value_curve(S,L,delta,wrap)

    # Get the index of the target with the given (max) value
    i = np.argmax(va)
    
    # return target that will give max ev and the actual ev
    return a[i], va[i]

def get_expected_values_and_optimal_action(S,L,delta,wrap=True):

    # Do convolution with resolution of "delta"
    va, a = compute_expected_value_curve(S,L,delta,wrap)

    # Get the index of the target with the given (max) value
    i = np.argmax(va)

    # return target that will give max ev and the actual ev
    # as well as all the other targets and evs (all the information from the convolution)
    return a,va,a[i],va[i]


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


    xskills = np.round(np.linspace(0.5,4.5,num=5),4)
    colors = ["tab:orange","tab:green","tab:red","tab:purple","tab:pink"]

    wrap = True


    numStates = 20

    states = generate_random_states(rng,3,5,numStates,0.5)

    for i in range(numStates):

        #'''
        fig = plt.figure()
        ax = plt.gca()
        #ax = plt.subplot2grid((5,2), (0, 0))

        plot_reward_profile(states[i], color = "tab:blue")

        plt.margins(0.10) 
        ax.autoscale(True)

        plt.ylim(-0.1,1.1)

        for j in range(len(xskills)):
            plot_expected_values(states[i],xskills[j],color=colors[j],label=xskills[j],wrap=wrap)


        plt.legend()

        plt.tight_layout()
        plt.savefig(folder+f"Plots-States{os.sep}1D-state" + str(i)+f"-EVs-Wrap{wrap}.png")
       
        plt.close()
        plt.clf()

        #'''

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
