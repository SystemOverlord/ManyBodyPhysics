# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':    

    N_weltl = 20
    N_points = 200
    
    weltl = np.array([[0]*N_points]*N_weltl)

    for i in xrange(N_weltl):
        for t in xrange(N_points):
            weltl[i][t] = i


    
    fig1 = plt.figure()
    fig1.show()
    for i in xrange(N_weltl):
        for t in xrange(N_points):
            plt.plot(weltl[i][t],t,'ro')
    
    
    