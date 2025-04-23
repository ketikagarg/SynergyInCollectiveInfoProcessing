import math
import numpy as np
import matplotlib.pyplot as plt

def distance(p):
    dist = 0
    # if diff:
    #     q = q[0]
    for i in range(len(p)):
        if p[i] != 1:
            dist += 1
    return dist


def hamming(x,y, xy, xy_pooled, num_potions):
    ##xy = x + y (Union of X and Y, excluding the new potions)
    ##xy_pooled = x + y + xy_pooled (Union of X and Y, and including the new potions)
    ##num_potions = number of all potions - is it 14? if we want to only consider the new potions, then need to change it to 7?
    c1 = (distance(x) + distance(y) - distance(xy)) / num_potions
    c2 = (distance(x) + distance(y) - distance(xy_pooled) ) / num_potions

    c3 = c1 - c2
##calculating seperately for c1 in order to calculate c3 has little meaning because they can only make one new potion,
##so c3 will always be 1/14, but whatever.
    return c1, c2, c3

