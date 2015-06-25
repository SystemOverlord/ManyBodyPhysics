# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy import weave

class walk_class:
    def __init__(self, walk_num):
        self.walk = walk_num
    
        if self.walk == 1: #rechts
            self.x = -0.5
            self.y = 0.5
        elif self.walk == 2: #unten
            self.x = 0.5
            self.y = 0.5
        elif self.walk == 3: #links
            self.x = 0.5
            self.y = -0.5
        elif self.walk == 4: #oben
            self.x = -0.5
            self.y = -0.5    

def Startconf(anzTeilchen,anzSpinUp,anzZeitschritte):
    
    weltlinien = np.array([[False]*anzTeilchen]*anzZeitschritte)    
    zahl = []
    zahl0 = random.randint(1,anzTeilchen-1)
    
    for k in range(anzSpinUp):
        while zahl0 in zahl:
            zahl0 = random.randint(1,anzTeilchen-1)

        zahl += [zahl0]
        weltlinien[:,zahl0] = True
    
    return weltlinien

def loop1(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien):
    
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
                
                # erhaltung der spinzahl
                L1 = (sum( weltlinien[0] ^ spinerhaltung_mask ) == anzSpinup)
                # gleichheit von start und endkonfiguration
                #L2 = np.array_equal(weltlinien[0], weltlinien[1])
                
                if L1: spinerhaltung = True

                break
          
        if weltlinienschnitt > 2 and spinerhaltung: break
            
    return x,y,l,k
       
def loop2(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien, Jz, beta):
    
    weltlinienschnitt = 0  
    spinerhaltung = False
    update = True

    mask1 = np.array([[0,0],[0,0]])
    mask2 = np.array([[1,1],[1,1]])
    mask3 = np.array([[1,0],[1,0]])
    mask4 = np.array([[0,1],[0,1]])
    mask5 = np.array([[1,0],[0,1]])
    mask6 = np.array([[0,1],[1,0]])
    
    walk_h = np.array([2, 1, 4, 3])
    walk_v = np.array([4, 3, 2, 1])
    
    w = np.tanh(Jz*beta/anzZeitschritte)
    
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
                x_u = x[k-1]+0.5
                x_o = x[k-1]-0.5
                y_r = y[k-1]+0.5
                y_l = y[k-1]-0.5
                
                # Periodische Randbedingungen        
                if x_u == anzZeitschritte: x_u = 0
                if x_o == -1: x_o = anzZeitschritte-1
                if y_r == anzTeilchen: y_r = 0
                if y_l == -1: y_l = anzTeilchen-1
            
                links_unten = weltlinien[x_u, y_l]
                rechts_unten = weltlinien[x_u, y_r]
                links_oben = weltlinien[x_o, y_l]
                rechts_oben = weltlinien[x_o, y_r]
                
                plakette = np.array([[links_oben, rechts_oben],[links_unten,rechts_unten]])
                
                if np.array_equal(plakette, mask1) or np.array_equal(plakette, mask2):
                    walk = walk_v[walkOld-1]
                    
                elif np.array_equal(plakette, mask3) or np.array_equal(plakette, mask4):
                    if w < random.random(): walk = walk_h[walkOld-1]
                    else: walk = walk_v[walkOld-1]
                    
                elif np.array_equal(plakette, mask5) or np.array_equal(plakette, mask6):
                    walk = walk_h[walkOld-1]
                    
                else: print 'Plakette verboten'
                
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
                
                # erhaltung der spinzahl
                L1 = (sum( weltlinien[0] ^ spinerhaltung_mask ) == anzSpinup)
                # gleichheit von start und endkonfiguration
                #L2 = np.array_equal(weltlinien[0], weltlinien[1])
                
                if L1: spinerhaltung = True

                break
          
        if weltlinienschnitt > 2 and spinerhaltung: break

        if m == 99: update = False  
        
    return x,y,l,k, update

def gewichter2(weltlinien):
    N = int(np.shape(weltlinien)[0])
    M = int(np.shape(weltlinien)[1])
    
    ns_py = np.array([0])
    nd_py = np.array([0])
    
    code = r'''
    int ns = 0;
    int nd = 0;
    
    for(int n = 0; n < N-1; n++) {
        for(int m = 0; m < M-1; m++) {
            
            if (n%2 != m%2) {
                int index1 = (n+1)*M + m;
                int index2 = (n+1)*M + m + 1;
                int index3 = (n)*M + m;
                int index4 = (n)*M + m + 1;
                
                int links_unten = weltlinien[index1];
                int rechts_unten = weltlinien[index2];
                int links_oben = weltlinien[index3];
                int rechts_oben = weltlinien[index4];
                
                if(links_unten == 1 & rechts_unten == 0 & links_oben == 1 & rechts_oben == 0) {
                    ns += 1;
                    }
                
                else if(links_unten == 0 & rechts_unten == 1 & links_oben == 0 & rechts_oben == 1) {
                    ns += 1;
                    }
            
                else if(links_unten == 1 & rechts_unten == 0 & links_oben == 0 & rechts_oben == 1) {
                    nd += 1;
                    }
                    
                else if(links_unten == 0 & rechts_unten == 1 & links_oben == 1 & rechts_oben == 0) {
                    nd += 1;
                    }
            }
        }    
    }
    ns_py[0] = ns;
    nd_py[0] = nd;

    '''

    weave.inline(code,['N','M','ns_py','nd_py','weltlinien'])
    
    return ns_py, nd_py


def autocorr(x):
    result = np.correlate(x, x, mode = 'full')
    maxcorr = np.argmax(result)
    #print 'maximum = ', result[maxcorr]
    result = result / result[maxcorr]     # <=== normalization

    return result[result.size/2:]


t0 = time.time()
# Parameter
anzTeilchen = 12
anzSpinup = 6
anzZeitschritte = 100
termination = anzTeilchen*anzZeitschritte
anzMarkovZeit = 10000
#m = anzTeilchen

Jz = 1.
Jx = 1.
beta = 0.01

#gs = np.exp(-Jz*beta/(2*anzZeitschritte)) * np.cosh(Jx*beta/anzZeitschritte)
#gd = np.exp(-Jz*beta/(2*anzZeitschritte)) * np.sinh(Jx*beta/anzZeitschritte)
gs_ratio = Jz/(anzZeitschritte) * (np.tanh(beta*Jx/anzZeitschritte) - 1) #gs' / gs
gd_ratio = Jz/(anzZeitschritte) * (1/np.tanh(beta*Jx/anzZeitschritte) - 1) #gd' / gd

meanNs = np.array([0]*anzMarkovZeit)
meanNd = np.array([0]*anzMarkovZeit)
energy = np.array([0.]*anzMarkovZeit)

# Start
weltlinien = Startconf(anzTeilchen,anzSpinup,anzZeitschritte)

meanNs[0] = anzTeilchen
meanNd[0] = 0
energy[0] = - meanNs[0]*Jz/(anzZeitschritte) * gs_ratio

# heat up array
for m in xrange(20):
    
    # Loop finden
    x,y,l,k = loop1(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien)

    # Spinflip
    for k in xrange(len(x)):
        weltlinien[int(x[k]),int(y[k])] = weltlinien[int(x[k]),int(y[k])] ^ True

# run simulation
for n in xrange(1,anzMarkovZeit):
    print n    
    
    # Loop finden
    x,y,l,k,update = loop2(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien, Jz, beta)

    # Spinflip
    if update:
        for k in xrange(len(x)):
            weltlinien[int(x[k]),int(y[k])] = weltlinien[int(x[k]),int(y[k])] ^ True 
        
        # Gewichter der Weltlinienkonfiguration berechnen   
        ns, nd = gewichter2(weltlinien)
        
        test_energy = -ns*gs_ratio - nd*gd_ratio
        test_energy /= (anzTeilchen*anzZeitschritte)
        
        meanNs[n] = int(ns)
        meanNd[n] = int(nd)
        energy[n] = test_energy
        

print(np.mean(energy[1000::]) - Jz/4)
#print(t0 - time.time())

#auto = autocorr(energy)
#
#figAuto = plt.figure()
#plt.plot(auto)
#plt.ylabel('Autocorrelation')
#plt.xlabel('Markov Time')

figEnergy = plt.figure()
plt.plot(energy)
plt.ylabel('Energy')
plt.xlabel('Markov Time')