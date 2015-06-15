# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time


def Startconf(anzTeilchen,anzSpinUp,anzZeitschritte):
    
    weltlinien = np.array([[False]*anzTeilchen]*anzZeitschritte)    
    zahl = []
    zahl0 = random.randint(1,anzTeilchen-1)
    
    for k in range(anzSpinup):
        while zahl0 in zahl:
            zahl0 = random.randint(1,anzTeilchen-1)

        zahl += [zahl0]
        weltlinien[:,zahl0] = True
    
    return weltlinien
       
def loop(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien):
    
    weltlinienschnitt = 0  
    spinerhaltung = False
    
    for m in xrange(100):
        
        x = np.array([0.0]*termination)              # Initialisierung
        y = np.array([0.0]*termination) 
        
        x[0] = random.randint(0,anzZeitschritte-1)
        y[0] = random.randint(0,anzTeilchen-1)  
        while weltlinien[x[0],y[0]] == 0:
            x[0] = random.randint(0,anzZeitschritte-1)
            y[0] = random.randint(0,anzTeilchen-1)
            
        breakvar = 0
        # Plakette links oben inaktiv
        if x[0]%2 == y[0]%2: walk = 1
        else: walk = 4
        
        walkOld = walk
    
        for k in xrange(1,termination):
            
            if k%2 == 0:  # Jeder zweite Schritt geht doppelt
           
                walk_array = []
                for walk in xrange(1,5):

                    # Verhindere Richtungsumkehr
                    if not ( (walkOld == 1 and walk == 3) or (walkOld == 3 and walk == 1) or (walkOld == 2 and walk == 4) or (walkOld == 4 and walk == 2) ):
                                
                        # switch für Schritt
                        if walk == 1: #rechts
                            x[k] = x[k-1]-0.5
                            y[k] = y[k-1]+0.5
                        elif walk == 2: #unten
                            x[k] = x[k-1]+0.5
                            y[k]= y[k-1]+0.5
                        elif walk == 3: #links
                            x[k] = x[k-1]+0.5
                            y[k] = y[k-1]-0.5
                        elif walk == 4: #oben
                            x[k] = x[k-1]-0.5
                            y[k] = y[k-1]-0.5    
    
                        # Periodische Randbedingungen        
                        if x[k] == anzZeitschritte: x[k] = 0
                        if x[k] == -1: x[k] = anzZeitschritte-1
                        if y[k] == anzTeilchen: y[k] = 0
                        if y[k] == -1: y[k] = anzTeilchen-1
                        
                        if (walk == 1 or walk == 4) and weltlinien[x[k],y[k]] == True: walk_array += [walk]
                        if (walk == 2 or walk == 3) and weltlinien[x[k],y[k]] == False: walk_array += [walk]

                if walk_array == []: break
                random.shuffle(walk_array)
                walk = walk_array[0]
                walkOld = walk 
            
            # switch für Schritt
            if walk == 1: #rechts
                x[k] = x[k-1]-0.5
                y[k] = y[k-1]+0.5
            elif walk == 2: #unten
                x[k] = x[k-1]+0.5
                y[k]= y[k-1]+0.5
            elif walk == 3: #links
                x[k] = x[k-1]+0.5
                y[k] = y[k-1]-0.5
            elif walk == 4: #oben
                x[k] = x[k-1]-0.5
                y[k] = y[k-1]-0.5
            
            # Periodische Randbedingungen        
            if x[k] == anzZeitschritte: x[k] = 0
            if x[k] == -1: x[k] = anzZeitschritte-1
            if y[k] == anzTeilchen: y[k] = 0
            if y[k] == -1: y[k] = anzTeilchen-1
            
            # Suche ob Loop sich beißt
            for l in xrange(0,k):
                    
                if (x[l]== x[k]) & (y[l] == y[k]):
                    x = x[l:k]
                    y = y[l:k]
                    breakvar = 1
                    break
            
            if breakvar ==  1:
                # Wenn Loop sich beißt, filtere Gitterpunkte raus
                if x[0] == np.floor(x[0]):
                    x = x[::2]
                    y = y[::2]
                else:
                    x = x[1::2]
                    y = y[1::2]          
                
                # überprüfe Anzahl der Schnittpunkte mit Weltlinie
                spinerhaltung_mask = np.array([False]*anzTeilchen)
                weltlinienschnitt = 0
                for n in xrange(len(x)):
                    if x[n] == 0: spinerhaltung_mask[y[n]] = True
                    if weltlinien[x[n],y[n]] == True: weltlinienschnitt += 1
                
                spinerhaltung = False
                
                if  sum( weltlinien[0] ^ spinerhaltung_mask ) == anzSpinup: spinerhaltung = True

                break
          
        if weltlinienschnitt > 2 and spinerhaltung: break
            
    return x,y,l,k


def gewichter(weltlinien, weight):

    mask1 = np.array([[1,0],[1,0]])
    mask2 = np.array([[0,1],[0,1]])
    mask3 = np.array([[1,0],[0,1]])
    mask4 = np.array([[0,1],[1,0]])
    #mask5 = np.array([[1,1],[1,1]])
    #mask6 = np.array([[0,0],[0,0]])
    
    ns = 0
    nd = 0
    weightConfig = 1
    for k in xrange(np.shape(weltlinien)[0]-1):
        for l in xrange(np.shape(weltlinien)[1]-1):
    
            if k%2 == l%2: # Gitterpunkt ist links unterer Punkt der aktiven Plakette
                links_unten = weltlinien[k,l]
                rechts_unten = weltlinien[k,l+1]
                links_oben = weltlinien[k+1,l]
                rechts_oben = weltlinien[k+1,l+1]
                
                configPlakette = np.array([[links_oben, rechts_oben],[links_unten,rechts_unten]])
                
                if np.array_equal(configPlakette, mask1): 
                    #weightConfig *= weight[0]
                    ns += 1
                elif np.array_equal(configPlakette, mask2): 
                    #weightConfig *= weight[1]
                    ns += 1
                elif np.array_equal(configPlakette, mask3): 
                    #weightConfig *= weight[2]
                    nd += 1
                elif np.array_equal(configPlakette, mask4): 
                    #weightConfig *= weight[3]
                    nd += 1
                #elif np.array_equal(configPlakette, mask5): weightConfig *= weight[4]
               # elif np.array_equal(configPlakette, mask6): weightConfig *= weight[5]    
                
    return weightConfig, ns, nd


t0 = time.time()
# Parameter
anzTeilchen = 51
anzSpinup = 6
anzZeitschritte = 100
termination = anzTeilchen*anzZeitschritte
anzMarkovZeit = 110
m = anzTeilchen

Jz = -1.
Jx = -1.
deltaTau = 1.
beta = 1.

meanNs = np.array([0]*anzMarkovZeit)
meanNd = np.array([0]*anzMarkovZeit)
energy = np.array([0]*anzMarkovZeit)

weight1 = np.exp(deltaTau*Jz/4)*np.cosh(deltaTau*Jx/2)
weight3 = -np.exp(deltaTau*Jz/4)*np.sinh(deltaTau*Jx/2)  
weight5 = np.exp(-deltaTau*Jz/4)  
weight = np.array([weight1, weight1, weight3, weight3, weight5, weight5])   

# Start
weltlinien = Startconf(anzTeilchen,anzSpinup,anzZeitschritte)

meanNs[0] = anzTeilchen
meanNd[0] = 0
energy[0] = - meanNs[0]*Jz/(2*m) *(np.tanh(beta*Jz/(2*m))-1)

akzeptanzrate = 0
T = 0
for n in xrange(1,anzMarkovZeit):
    # Loop finden
    x,y,l,k = loop(anzZeitschritte, anzTeilchen, anzSpinup, termination,weltlinien)

    weltlinienOld = weltlinien 
    # Spinflip
    for k in xrange(len(x)):
        weltlinien[int(x[k]),int(y[k])] = weltlinien[int(x[k]),int(y[k])] ^ True 
        
    # Gewichter der Weltlinienkonfiguration berechnen   
    weightConfig, ns, nd = gewichter(weltlinien, weight) 
    test_meanNs = ns
    test_meanNd = nd
     
    test_energy = - test_meanNs*Jz/(2*m) *(np.tanh(beta*Jz/(2*m))-1) - test_meanNd*Jz/(2*m) *(1/np.tanh(beta*Jz/(2*m))-1) #- Jz*anzTeilchen/4  (kürzt sich in MC)
    
    if exp(-beta*(test_energy-energy[n-1])) < np.random.rand(1):
        weltlinien = weltlinienOld
        meanNs[n] = meanNs[n-1]
        meanNd[n] = meanNd[n-1]
        energy[n] = energy[n-1]
    else:
        akzeptanzrate += 1
        meanNs[n] = test_meanNs
        meanNd[n] = test_meanNd
        energy[n] = test_energy    
        

energyMean = - sum(meanNs)/anzMarkovZeit*Jz/(2*m) *(np.tanh(beta*Jz/(2*m))-1) - sum(meanNd)/anzMarkovZeit*Jz/(2*m) *(1/np.tanh(beta*Jz/(2*m))-1) #- Jz*anzTeilchen/4  (kürzt sich in MC)
print(energyMean, np.mean(energy))
