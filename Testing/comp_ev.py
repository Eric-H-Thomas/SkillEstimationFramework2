# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 00:50:14 2022

@author: User1
"""

from dartboard import *
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
from matplotlib import pyplot as plt

import code


#Get all of the possible actions in our domain
def getActions():

    '''
    actions = [(0,0)] #center

    #Distances are inner-single, triple, outer-single, double
    distances = [66, 103, 135, 166]
    angle_increment = math.pi / 10.0
    angle = 0

    for wedge in range(20):
        for d in distances:
            #Get (x,y) location for this target)
            x = d*math.cos(angle)
            y = d*math.sin(angle)
            actions.append((x,y))
        angle += angle_increment
    '''


    # '''
    allTargets = []

    defaultX = np.arange(-170.0,171.0,5.0)
    defaultY = np.arange(-170.0,171.0,5.0)

    for xi in defaultX:
        for yi in defaultY:
            allTargets.append((xi,yi))

    actions = np.array(allTargets)
    # '''

    
    # code.interact("...", local=dict(globals(), **locals()))


    return actions


def plotValueFunction(value,skill):
    plt.scatter(list(range(len(value))), value)
    plt.title("Skill level of " + str(np.sqrt(skill)))
    plt.show()

def get_actions_dense(min_arc_length = 2, min_r_step = 2):
    alist = []
    alist.append([0.0,0.0])
    rs = [i for i in range(16,99,min_r_step)]
    rs += [ i for i in range(99,162,min_r_step)]
    rs += [i for i in range(162,171, min_r_step)]
    slices = [np.pi/10 * i + np.pi/20 for i in range(20)]

    
    for r in rs:
        t_per_slice = get_t_per_slice(r,min_arc_length)
        for t in slices:

            for delta_t in t_per_slice:
                
                alist.append([r,t+delta_t])

    alist = np.array(alist)


    x = alist[:,0] * np.cos(alist[:,1])
    y = alist[:,0] * np.sin(alist[:,1])
    
    return np.vstack((x,y)).T


def new_vi(start_score, xskill, actions, singleProbs, doubleProbs,rules = 0, vguess = None):
    
    # code.interact("Before vi...", local=dict(globals(), **locals()))

    
    if vguess is None:
        value = -1 * np.linspace(np.sqrt(xskill),1.5*np.sqrt(xskill) ,start_score+1)
    else:
        value = vguess.copy()
    value[0] = 0.0 #Value of state 0 (we are done!)
    value[1] = 0.0  #value of state 1 (never get here!)
    delta = 10.0 #How much did it change this iter?
    gamma = 1.0 #Discount factor
    tolerance = 0.001* np.sqrt(xskill) #When do we stop?
    resolution = 5.0
    iters = 0

    PI = [[None,None]] * (startScore+1)
    MAX_EV = [None] * (startScore+1)


    while delta > tolerance:
        delta = 0.0 #reset delta

        if rules == 0:
            for s in range(2,len(value)):
                if s <61:
                    score_change = singleProbs[:,:s-1] @ np.flip(value[2:s+1])
                    bust = np.sum(singleProbs[:,s-1:],axis=1) * value[s]
                    
                    doub_change = doubleProbs[:,:s-1] @ np.flip(value[2:s+1])
                    doub_bust = (np.sum(doubleProbs[:,s+1:],axis=1) + doubleProbs[:,s-1])  * value[s]
                    
                    EV = score_change+bust+doub_change+doub_bust
                    bestEV = np.max(EV)
                    
                else:
                    score_change = (singleProbs + doubleProbs) @ np.flip(value[s-60:s+1])
                    EV = score_change
                    bestEV = np.max(EV)
                    
                            
                # current_delta = abs(value[s] + 1 - gamma*bestEV) 
                current_delta = abs(value[s] + 1 - gamma*bestEV) 
                if current_delta > delta:
                    delta = current_delta
    
                #Update the value
                value[s] = -1 + gamma*bestEV    

                mi = np.unravel_index(EV.argmax(), EV.shape)[0]
                action = [actions[mi][0],actions[mi][1]]
                PI[s] = action
                MAX_EV[s] = bestEV


        elif rules == 1:
            Probs = singleProbs+doubleProbs
            for s in range(2,len(value)):
                if s <61:
                    change = Probs[:,:s-1] @ np.flip(value[2:s+1])
                    bust = (np.sum(Probs[:,s+1:],axis=1) + Probs[:,s-1])  * value[s]
                    
                    bestEV = np.max(change+bust)
                    
                    
                else:
                    score_change = Probs @ np.flip(value[s-60:s+1])
                    bestEV = np.max(score_change)
                    
                            
                current_delta = abs(value[s] + 1 - gamma*bestEV) 
                if current_delta > delta:
                    delta = current_delta
    
                #Update the value
                value[s] = -1 + gamma*bestEV  
            
        elif rules == 2:
            Probs = singleProbs+doubleProbs
            for s in range(1,len(value)):
                if s < 61:
                    change = Probs[:,:s] @ np.flip(value[1:s+1])
                    bestEV = np.max(change)
                    
                    
                else:
                    score_change = Probs @ np.flip(value[s-60:s+1])
                    bestEV = np.max(score_change)
                    
                            
                current_delta = abs(value[s] + 1 - gamma*bestEV) 
                if current_delta > delta:
                    delta = current_delta
    
                #Update the value
                value[s] = -1 + gamma*bestEV
        
        print("Current Delta: ", delta)
        # code.interact(f"after iter {iters}...", local=dict(globals(), **locals()))
        iters += 1


    print('Value Iteration Converged after', iters, 'iterations')
    print('Final Value Function')
    plotValueFunction(value,xskill)
    # code.interact("...", local=dict(globals(), **locals()))
    
    return value,iters,MAX_EV,PI


def precalc(mus,var,sample_size=100000):
    N = multivariate_normal([0,0], var)
    ps = N.rvs(size = sample_size)

    
    non_doubs_ld = np.zeros((mus.shape[0],61),dtype=np.longdouble)
    doub_a_ld = np.zeros_like(non_doubs_ld,dtype=np.longdouble)

    non_doubs_f = np.zeros((mus.shape[0],61),dtype=np.float64)
    doub_a_f = np.zeros_like(non_doubs_f,dtype=np.float64)


    for i,mu in enumerate(mus):
        
        p = ps+mu
        ss,doubs = npscore(p[:,0],p[:,1],return_doub=True)
        # code.interact("npscore...", local=dict(globals(), **locals()))
        
        nond, nonc = np.unique(ss[~doubs], return_counts=True)
        x = np.sum(nonc)
        doub,c = np.unique(ss[doubs], return_counts=True)
        y = np.sum(c)

        non_doubs_ld[i,nond] = nonc/sample_size
        doub_a_ld[i,doub] = c/sample_size

        non_doubs_f[i,nond] = nonc/sample_size
        doub_a_f[i,doub] = c/sample_size


    # code.interact("...", local=dict(globals(), **locals()))
        
    return non_doubs_ld, doub_a_ld


def get_t_per_slice(radius,mind):
    ssize = radius * np.pi/10
    
    num_points = int(np.ceil(ssize/mind) )
    if num_points %2 ==0:
        num_points +=1
        
    thetas = np.linspace(0,np.pi/10, num= num_points, endpoint=False)
    return thetas
    


if __name__ == "__main__":

    np.random.seed(0)

    i_vals = []
    startScore = 201 #501
    numSamples = 100_00
    # xSkills = [i**2 for i in np.linspace(5,170,165)]
    xSkills = [2.5**2] 
    # xSkills = [150.5**2] 


    actions = getActions()
    #actions = get_actions_dense(3)
    
    # initial guesses
    double2 = -1 * np.linspace(1,16 ,startScore+1)
    no_double2 = -1 * np.linspace(1,16,startScore+1)
    no_bust = -1 * np.linspace(1,16 ,startScore+1)
    skill_dict = {}
    
    
    for i in tqdm(range(len(xSkills))):
        xSkill = xSkills[i]
        singleProbs, doubleProbs = precalc(np.array(actions), xSkill, numSamples)
        #try for all rule sets
        double2,i1, MAX_EV, PI = new_vi(startScore, xSkill, actions, singleProbs, doubleProbs,vguess=double2)
        # no_double2,i2 =  new_vi(startScore, xSkill, actions, singleProbs, doubleProbs, rules=1,vguess=no_double2)
        # no_bust,i3 = new_vi(startScore, xSkill, actions, singleProbs, doubleProbs, rules=2,vguess=no_bust)
        
        # skill_dict[i+5] = np.column_stack([double2,no_double2,no_bust])
        # i_vals.append(i1+i2+i3)
        
    # plt.plot(i_vals)
    # plt.show()

    code.interact("...", local=dict(globals(), **locals()))
