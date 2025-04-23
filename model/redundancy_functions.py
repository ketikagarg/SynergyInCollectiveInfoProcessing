# author: KG - 6 Jan 2024 
# purpose: sharing function to calculate redundancy and synergy for different levels of trajectory: whole, A, B 

import numpy as np


_dict = {} # dictionary to store  values for each pair of agents

def prob(x, all) :
    ## this function calculates probability of each potion in a set
    ## x is the binary vector for potions - [0,1,1,0]
    ## all is the set of the possible values of potions, currently - [0,1]
    c = [] 
    for a in all:
        c.append(x.count(a))
    return np.array(c) 


def joint_prob(x,y,w, all, num ): 
    ## this function calculates joint probability between 2 or 3 sets
    ##num is the number of sets for which joint prob is calculated

    if num == 2: ##for the case of joint_prob between 2 sets
        ##create empty matrix             
        joint_prob = np.zeros((len(all), len(all)))
        for i, xx in enumerate(all):
            for j, yy in enumerate(all):
                joint_prob[i, j] = np.sum((np.array(x) == xx) & (np.array(y) == yy)) / len(x)

    else : ##for the case of joint_prob between 3 sets
            ##create empty matrix
        joint_prob = np.zeros((len(all), len(all), len(all)))
        for i, xx in enumerate(all):
            for j, yy in enumerate(all):
                for k, ww in enumerate(all):
                    joint_prob[i, j, k] = np.sum((np.array(x) == xx) & (np.array(y) == yy) & (np.array(w) == ww)) / len(x)
    
    return joint_prob




def redundancy(X, Y, W):
     ##X, Y are potion sets from connected neighbors 
    ## W is the combined set of potions from X and Y (where new potions can be added to W)
     ## t is the time step
     ## x_num and y_num are indices of X,Y agents for storage
    ## type = 0 for the case of considering whole, type = 1 for the case of 3 agents

    all_potions = [1,0]
    # all_potions = np.array(list(set(X + Y + W))) # set( X + Y + W )


    ##calculate probabilities of each potion in each set
    p_x = prob(X, all_potions) / len(X)
    p_y = prob(Y, all_potions) / len(Y)
    p_w = prob(W, all_potions) / len(W)
    #print(p_x, p_y)

    ##create joint probability matrices 
    ## between different sets
    joint_xy = joint_prob(X,Y,W, all_potions, 2)
    joint_xw = joint_prob(X,W,W, all_potions, 2)
    joint_yw = joint_prob(Y,W, W, all_potions, 2)
    joint_xyw = joint_prob(X,Y, W,all_potions , 3)

    ##caluclate MI 
    mi = 0 
    for i in range(len(all_potions)):
        for j in range(len(all_potions)):
            if joint_xy[i, j] > 0:

                mi += joint_xy[i, j] * np.log2( (joint_xy[i, j]) / (p_x[i] * p_y[j]) )


    ##calculate conditional mi 
    cmi = 0 
    for i in range(len(all_potions)):
        for j in range(len(all_potions)):
            for k in range(len(all_potions)):
                if joint_xyw[i, j, k] > 0:
                    _n = p_w[k] * joint_xyw[i,j,k]
                    _d = joint_xw[i, k] * joint_yw[j,k]                  
                    cmi += joint_xyw[i,j,k] * np.log2(_n / _d )

    ##calculate redundancy
    red = mi - cmi
    return mi, red, cmi


def calculate_redundancy(X, Y, W, all_potions, t, x_num, y_num):
    # Call redundancy function for the case of considering the whole vectors
    mi, red, cmi = redundancy(X, Y, W, all_potions)
    ##store mi and redundancy value for the pair X,Y into the dictionary
    ## it will store values for each time step and for each type of level (whole, A, B)
    _dict[x_num][y_num]['whole'][t]['mi'] = mi
    _dict[x_num][y_num]['whole'][t]['red'] = red
    _dict[x_num][y_num]['whole'][t]['red'] = cmi

    # Call redundancy function for the case of considering only the first half of the vectors
    half_X = X[:len(X)//2]
    half_Y = Y[:len(Y)//2]
    half_W = W[:len(W)//2]
    mi, red, cmi = redundancy(half_X, half_Y,half_W, all_potions)
    _dict[x_num][y_num]['A'][t]['mi'] = mi
    _dict[x_num][y_num]['A'][t]['red'] = red
    _dict[x_num][y_num]['A'][t]['red'] = cmi

    # Call redundancy function for the case of considering only the second half of the vectors
    half_X = X[len(X)//2:]
    half_Y = Y[len(Y)//2:]
    half_W = W[len(W)//2:]
    mi, red, cmi = redundancy(half_X, half_Y, half_W, all_potions)
    _dict[x_num][y_num]['B'][t]['mi'] = mi
    _dict[x_num][y_num]['B'][t]['red'] = red
    _dict[x_num][y_num]['B'][t]['red'] = cmi



    
    



    
    