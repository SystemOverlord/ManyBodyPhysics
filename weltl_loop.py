# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def Startconf(anzTeilchen,anzSpinUp,anzZeitschritte):
    
    weltlinien = np.array([[False]*anzTeilchen]*anzZeitschritte)    
    zahl = []
    zahl0 = random.randint(1,anzTeilchen-1)
    
    for k in range(anzSpinup-1):
        while zahl0 in zahl:
            zahl0 = random.randint(1,anzTeilchen-1)

        zahl += [zahl0]
        weltlinien[:,zahl0] = True
    
    return weltlinien
       
    

def loop(anzZeitschritte, anzTeilchen,termination, weltlinien):
    x = np.array([0.0]*termination)              # Initialisierung
    y = np.array([0.0]*termination) 
    
    x[0] = random.randint(0,anzTeilchen-1)
    y[0] = random.randint(0,anzZeitschritte-1)  
    while weltlinien[x[0],y[0]] == 0:
        x[0] = random.randint(0,anzTeilchen-1)
        y[0] = random.randint(0,anzZeitschritte-1)
            
    breakvar = 0

    if x[0]%2 == 0: walk = 1
    else: walk = 4
    
    walkOld = walk

    for k in range(1,termination): 
        print(k)
        if k%2 == 0:  # Jeder zweite Schritt geht doppelt
            walk = random.randint(1,4)
            
            breakvar1 = True
            
            while breakvar1:
                
                walk = random.randint(1,4)
                
                # Verhindere Richtungsumkehr
                if not ( (walkOld == 1 and walk == 3) or (walkOld == 3 and walk == 1) or (walkOld == 2 and walk == 4) or (walkOld == 4 and walk == 2) ):
                    breakvar1 = False
                        
                # überprüft of schrittrichtungssinn der Pfeilfolge folgt
                if breakvar1 == False:
                    
                    # switch für Schritt
                    if walk == 1: #rechts
                        x[k] = x[k-1]+0.5
                        y[k] = y[k-1]+0.5
                    elif walk == 2: #unten
                        x[k] = x[k-1]+0.5
                        y[k]= y[k-1]-0.5
                    elif walk == 3: #links
                        x[k] = x[k-1]-0.5
                        y[k] = y[k-1]-0.5
                    elif walk == 4: #oben
                        x[k] = x[k-1]-0.5
                        y[k] = y[k-1]+0.5    

                    # Periodische Randbedingungen        
                    if x[k] == anzTeilchen:
                        x[k] = 0
                    if x[k] == -1:
                        x[k] = anzTeilchen-1
                    if y[k] == anzZeitschritte:
                        y[k] = 0
                    if y[k] == -1:
                        y[k] = anzTeilchen-1
                    print(weltlinien[x[k],y[k]], walk)
                    if (walk == 1 or walk == 4) and weltlinien[x[k],y[k]] == False: breakvar1 = True
                    if (walk == 2 or walk == 3) and weltlinien[x[k],y[k]] == True: breakvar1 = True

            walkOld = walk
        
        else:
            # switch für Schritt
            if walk == 1: #rechts
                x[k] = x[k-1]+0.5
                y[k] = y[k-1]+0.5
            elif walk == 2: #unten
                x[k] = x[k-1]+0.5
                y[k]= y[k-1]-0.5
            elif walk == 3: #links
                x[k] = x[k-1]-0.5
                y[k] = y[k-1]-0.5
            elif walk == 4: #oben
                x[k] = x[k-1]-0.5
                y[k] = y[k-1]+0.5

        
        # Suche ob Loop sich beißt
        for l in range(0,k):
            if (x[l]== x[k]) & (y[l] == y[k]):                
                print('break',l)
                x = x[l:k]
                y = y[l:k]
                breakvar = 1
                break
            
        if breakvar ==  1: 
            return x,y,l,k
            break
    

        
    return x,y,l,k                



anzTeilchen = 5
anzSpinup = 3
anzZeitschritte = 5
termination = 100

weltlinien = Startconf(anzTeilchen,anzSpinup,anzZeitschritte)


print(weltlinien)
x,y,l,k = loop(anzZeitschritte, anzTeilchen, termination,weltlinien)

print(x)
#weltlinien[x[::2],y[::2]] = weltlinien[x[::2],y[::2]] | True
print(weltlinien)


#print('x', x[l:k+1])
#print('y', y[l:k+1])
#print('l= ',l, 'k = ',k)

#fig = plt.figure()
#plt.plot(x[l:k],y[l:k])
#plt.grid()
#plt.xlim(-1, anzTeilchen)
#plt.ylim(-1,anzZeitschritte)
