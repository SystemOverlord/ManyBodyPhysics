# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def Startconf(anzTeilchen,anzSpinUp):
    conf = np.array([bool(0)]*anzTeilchen)
    i = 0
    while i < anzSpinUp:
        tmp = random.randint(0, anzTeilchen-1)
        if conf[tmp] == 0:
            conf[tmp] = 1
            i = i+1
    return conf
    
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
        
    

def loop(anzZeitschritte, anzTeilchen,termination, weltlinien):
    x = np.array([0]*termination)              # Initialisierung
    y = np.array([0]*termination) 
    x[0] = random.randint(0,anzTeilchen-1)
    y[0] = random.randint(0,anzZeitschritte-1)     
    breakvar = 0
    
    walk = random.randint(1,4)
    walkOld = walk

    for k in range(1,termination): 
        if k%2 == 0:  # Jeder zweite Schritt geht doppelt
            walk = random.randint(1,4)
            while (walkOld + walk == 4) | (walkOld +walk == 6) :    # Verhindere Richtungsumkehr
                walk = random.randint(1,4)

            walkOld = walk 
        
        # switch für Schritt
        if walk == 1: #rechts
            x[k] = x[k-1]+1
            y[k] = y[k-1]
        elif walk == 2: #unten
            x[k] = x[k-1]
            y[k]= y[k-1]-1
        elif walk == 3: #links
            x[k] = x[k-1]-1
            y[k] = y[k-1]
        elif walk == 4: #oben
            x[k] = x[k-1]
            y[k] = y[k-1]+1
                
        # Periodische Randbedingungen        
        if x[k] == anzTeilchen:
            x[k] = 0
        if x[k] == -1:
            x[k] = anzTeilchen-1
        if y[k] == anzZeitschritte:
            y[k] = 0
        if y[k] == -1:
            y[k] = anzTeilchen-1
        
        # Suche ob Loop sich beißt
        for l in range(0,k):
            if (x[l]== x[k]) & (y[l] == y[k]):                
                print('break',l)
                breakvar = 1
                break
            
        if breakvar ==  1: 
            return x,y,l,k
            break
    
        k += 1

        
    return x,y,l,k                



anzTeilchen = 10
anzZeitschritte = 10
termination = 100
x,y,l,k = loop(anzZeitschritte, anzTeilchen, termination,1)
print('x', x[l:k+1])
print('y', y[l:k+1])
print('l= ',l, 'k = ',k)

fig = plt.figure()
plt.plot(x[l:k],y[l:k])
plt.grid()
plt.xlim(-1, anzTeilchen)
plt.ylim(-1,anzZeitschritte)




if __name__ == '__main__':    

    anzTeilchen = 5       
    anzSpinUp = 2
    anzZeitschritte = 4   # muss durch 2 teilbar sein
    
    gitter = np.array([Startconf(anzTeilchen,anzSpinUp)]*anzZeitschritte)
    indizesXY, indizesBereinigt = FeldindizesXY(gitter)
    
    
    
#    #============== Plot Felder =================
#    fig=plt.figure()
#    for i in xrange(np.shape(indizesBereinigt)[0]):    
#        plt.plot(indizesBereinigt[i][:,1],indizesBereinigt[i][:,0])   
#    plt.xlim(-1,anzTeilchen)
#    plt.ylim(-1,anzZeitschritte)



    
    