# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt
import random

    
def FeldindizesXY(gitter):
    xmax = np.shape(gitter)[0]-1
    ymax = np.shape(gitter)[1]-1
    indizesXY = np.array([[[[0]*2]*4]*xmax]*ymax)
    
    for y in xrange(ymax):
        for x in xrange(xmax):
            #print('x:',x,'/',xmax,'y:',y,'/',ymax,'shape:',np.shape(indizesXY))
            indizesXY[y][x][0][0] = x     # oben links X
            indizesXY[y][x][0][1] = y+1   # oben links Y
            indizesXY[y][x][1][0] = x+1   # oben rechts X
            indizesXY[y][x][1][1] = y+1   # oben rechts Y
            indizesXY[y][x][2][0] = x     # unten links X
            indizesXY[y][x][2][1] = y     # unten links Y       
            indizesXY[y][x][3][0] = x+1   # unten rechts X
            indizesXY[y][x][3][1] = y     # unten rechts Y

            if x == 0 and y == 0:
                indizesBereinigt = [indizesXY[y][x]]
            if (y%2 == x%2) and not (x==0 and y==0): 
                #print np.shape(indizesXY[y][x]),np.shape(indizesBereinigt)
                indizesBereinigt = np.concatenate((indizesBereinigt,[indizesXY[y][x]]))
    return indizesXY, indizesBereinigt
    
def Gitterplot():
    fig1 = plt.figure()
    fig1.show()
    for t in xrange(anzZeitschritte):
        for i in xrange(anzTeilchen):
            if gitter[t][i]:
                plt.plot(i,t,'ro')    

def Startconf(anzTeilchen,anzSpinUp):
    conf = np.array([bool(0)]*anzTeilchen)
    i = 0
    while i < anzSpinUp:
        tmp = random.randint(0, anzTeilchen-1)
        if conf[tmp] == 0:
            conf[tmp] = 1
            i = i+1
    
    return conf
          
                        



if __name__ == '__main__':    

    anzTeilchen = 5       
    anzSpinUp = 2
    anzZeitschritte = 4   # muss durch 2 teilbar sein
    
    gitter = np.array([Startconf(anzTeilchen,anzSpinUp)]*anzZeitschritte)
    indizesXY, indizesBereinigt = FeldindizesXY(gitter)
    

    
    
    #============== Plot Felder =================
    fig=plt.figure()
    for i in xrange(np.shape(indizesBereinigt)[0]):    
        plt.plot(indizesBereinigt[i][:,1],indizesBereinigt[i][:,0])   
    plt.xlim(-1,anzTeilchen)
    plt.ylim(-1,anzZeitschritte)



    
    