import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np

def wrap_angle_360(angle):
    while angle > 2*math.pi:
        angle -= 2*math.pi
    while angle < 0.0:
        angle += 2*math.pi

    return angle

slices = np.array([11,8,8,16,16,7,7,19,19,3,3,17,17,2,2,15,15,10,10,6,6,13,13,4,4,18,18,1,1,20,20,5,5,12,12,9,9,14,14,11,11])


def npscore(x,y, return_doub=False):
    a = np.arctan2(y,x)
    r = np.hypot(x,y)
    

    a = ((a + math.pi) * (40/(2*math.pi))).astype(int)
    ans = slices[a]
    
    
    doubb = r <6.35
    bulls = r<15.9
    on = r<170
    
    trip = (r>99) & (r< 107)
    doub = (r<170) & (r>162)
    

    ans *= np.invert(bulls) *on 
    ans += (25*bulls) + ( 25*doubb)
    ans += (ans*trip * 2) + (ans*doub)
    
    if return_doub:
        return ans, doub+doubb
    return ans 
    
    
def testscoretime(x,y):
    for i in range(len(x)):
        score(x[i],y[i])

        
    
    
def score(x,y):
    """ 
    Return the score for location (x,y) on the dartboard. 
    Units are mm    
    """
    #First convert to polar coordinates to get distance from (0,0) and angle from (0,0)
    a = math.atan2(y,x) #angle
    r = math.hypot(x,y) #radius

    scaling = 1.0

    #Figure out which distance we fall in 
    if r < 6.35:
        #Double bullseye = 50 points
        return 50.0
    elif r < 15.9: 
        #Single bullseye = 25 points
        return 25.0

    elif r > 99 and r < 107:
        # Triple score
        scaling = 3.0

    elif r > 162 and r < 170:
        # Double score
        scaling = 2.0

    elif r > 170:
        # Off the board
        return 0.0
    #slices = [11,8,8,16,16,7,7,19,19,3,3,17,17,2,2,15,15,10,10,6,6,13,13,4,4,18,18,1,1,20,20,5,5,12,12,9,9,14,14,11]

    #a = int((a + math.pi) * (40/(2*math.pi))) 

    #return scaling*slices[a]


    #Figure out which slice we fall in
    angle_increment = math.pi / 10.0
    slice_low = - math.pi - angle_increment / 2.0
    slice_high = slice_low + angle_increment

    slices = [11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]

    for i in range(21):
        if a > slice_low and a < slice_high:
            return scaling*slices[i]
        slice_low += angle_increment
        slice_high += angle_increment
    
    
    
    #Check for 11 slice

    #Must have missed the board!
    return 0.0


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
    label_regions()

def label_regions():
    slices = [11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]

    angle_increment = math.pi / 10.0    
    angle = math.pi
    r = 190.0

    for i in range(20):
        x = r*math.cos(angle)
        y = r*math.sin(angle)
        plt.text(x,y,str(slices[i]),fontsize=12,horizontalalignment='center')
        angle += angle_increment


if __name__ == "__main__":
    print("Displaying dartboard . . .\n")
    plt.figure()
    ax = plt.gca()
    draw_board(ax)
    label_regions()    
    plt.axis('equal')

    plt.show()





