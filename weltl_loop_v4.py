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

def Startconf(anzTeilchen,anzSpinUp,anzZeitschritte):
    
    weltlinien = np.array([[False]*anzTeilchen]*anzZeitschritte)    
    zahl = []
    zahl0 = random.randint(1,anzTeilchen-1)
    
#    weltlinien[:,::2] = True
    
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

def walk_neu(anzZeitschritte, anzTeilchen, weltlinien, x, y, k, walkOld, gs, gd, beta):

    #w = np.tanh(Jz*beta/anzZeitschritte)
    #w = np.exp(beta * (gs_ratio - gd_ratio) )
    
    mask1 = np.array([[0,0],[0,0]])
    mask2 = np.array([[1,1],[1,1]])
    mask3 = np.array([[1,0],[1,0]])
    mask4 = np.array([[0,1],[0,1]])
    mask5 = np.array([[1,0],[0,1]])
    mask6 = np.array([[0,1],[1,0]])
    
    walk_h = np.array([2, 1, 4, 3])
    walk_v = np.array([4, 3, 2, 1])
    
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
                
    # Plaketten vom Typ 1+, 1- (siehe Everz S9)
    if np.array_equal(plakette, mask1) or np.array_equal(plakette, mask2):
        if random.random() < gs / (gs + gd): walk = walk_v[walkOld-1]
        else: walk = walkOld
                
    # Plaketten vom Typ 2+, 2- (siehe Everz S9)
    elif np.array_equal(plakette, mask3) or np.array_equal(plakette, mask4):
        if random.random() < 1./(1.+gd): walk = walk_v[walkOld-1]
        else: walk = walk_h[walkOld-1]
                
    # Plaketten vom Typ 3+, 3- (siehe Everz S9)
    elif np.array_equal(plakette, mask5) or np.array_equal(plakette, mask6):
        if random.random() < 1./(1.+gs): walk = walkOld
        else: walk = walk_h[walkOld-1]
                    
    else:
        print 'Plakette verboten'
        exit(1)
        
    return walk

def walk_neu2(anzZeitschritte, anzTeilchen, weltlinien, x, y, k, walkOld, gs, gd, beta):

    #w = np.tanh(Jz*beta/anzZeitschritte)
    #w = np.exp(beta * (gs_ratio - gd_ratio) )
    
    mask1 = np.array([[0,0],[0,0]])
    mask2 = np.array([[1,1],[1,1]])
    mask3 = np.array([[1,0],[1,0]])
    mask4 = np.array([[0,1],[0,1]])
    mask5 = np.array([[1,0],[0,1]])
    mask6 = np.array([[0,1],[1,0]])
    
    walk_h = np.array([2, 1, 4, 3])
    walk_v = np.array([4, 3, 2, 1])
    
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
                
    # Plaketten vom Typ 1+, 1- (siehe Everz S9)
    if np.array_equal(plakette, mask1) or np.array_equal(plakette, mask2):
        walk = walk_v[walkOld-1]
                
    # Plaketten vom Typ 2+, 2- (siehe Everz S9)
    elif np.array_equal(plakette, mask3) or np.array_equal(plakette, mask4):
        if random.random() < np.tanh(beta/anzZeitschritte): walk = walk_h[walkOld-1]
        else: walk = walk_v[walkOld-1]
                
    # Plaketten vom Typ 3+, 3- (siehe Everz S9)
    elif np.array_equal(plakette, mask5) or np.array_equal(plakette, mask6):
        walk = walk_h[walkOld-1]
                    
    else:
        print 'Plakette verboten'
        exit(1)
        
    return walk
   
def loop2(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien, Jz, beta, gs, gd):
    
    weltlinienschnitt = 0  
    spinerhaltung = False
    update = True
        
    for m in xrange(100):
        
        x = np.array([0.0]*termination)              # Initialisierung
        y = np.array([0.0]*termination) 
        
        x[0] = random.randint(0,anzZeitschritte-1)
        y[0] = random.randint(0,anzTeilchen-1)  
        while weltlinien[x[0],y[0]] == False:
            x[0] = random.randint(0,anzZeitschritte-1)
            y[0] = random.randint(0,anzTeilchen-1)
            
        breakvar = 0
        # Plakette links oben inaktiv
        if x[0]%2 == y[0]%2: walk = 1
        else: walk = 4
        
        for k in xrange(1,termination):
            
            if k%2 == 0:  # Jeder zweite Schritt geht doppelt
                walk = walk_neu(anzZeitschritte, anzTeilchen, weltlinien, x, y, k, walk, gs, gd, beta)
            
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
            else: #oben
                x[k] = x[k-1]-0.5
                y[k] = y[k-1]-0.5
            
            #TODO: optimieren: class xy
            # Periodische Randbedingungen        
            if x[k] == anzZeitschritte: x[k] = 0
            if x[k] == -1: x[k] = anzZeitschritte-1
            if y[k] == anzTeilchen: y[k] = 0
            if y[k] == -1: y[k] = anzTeilchen-1
            
            #TODO: sort + bisec 
            # Suche ob Loop sich beißt
            for l in xrange(0,k-1):
                    
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
                    # sucht nach Loop-Schnittpunkten in der ersten Zeile (x[n] == 0) des Schachbretts
                    if x[n] == 0: spinerhaltung_mask[y[n]] = True
                    
                    # sucht Schnittpunkte mit Weltlinien
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

def gewichter(weltlinien):
    weltlinien_neu = np.hstack((weltlinien, weltlinien[:,0].reshape(-1, 1)))
    weltlinien_neu = np.vstack((weltlinien, weltlinien[0,:]))
    
    N = int(np.shape(weltlinien_neu)[0])
    M = int(np.shape(weltlinien_neu)[1])
    
    ns_py = np.array([0])
    nd_py = np.array([0])
    n0_py = np.array([0])
    val_py= np.array([0])
    
    code = r'''
    int ns = 0;
    int nd = 0;
    int n0 = 0;
    int val = 0;
    
    for(int n = 0; n < N-1; n++) {
        for(int m = 0; m < M-1; m++) {
            
            if (n%2 != m%2) {
                int index1 = (n+1)*M + m;
                int index2 = (n+1)*M + m + 1;
                int index3 = n*M + m;
                int index4 = n*M + m + 1;
                
                int links_unten = weltlinien_neu[index1];
                int rechts_unten = weltlinien_neu[index2];
                int links_oben = weltlinien_neu[index3];
                int rechts_oben = weltlinien_neu[index4];
                
                int valide = 0;
                if(links_unten == links_oben && rechts_oben == rechts_unten && rechts_unten != links_oben) {
                    ns += 1;
                    valide +=1;
                    }
                if(links_unten != links_oben && rechts_oben != rechts_unten && rechts_unten == links_oben) {
                    nd += 1;
                    valide +=1;
                    }
                if(links_unten == links_oben && rechts_oben == rechts_unten && rechts_unten == links_oben) {
                    n0 += 1;
                    valide +=1;
                    }
                if(valide!=1){val+=1;}
                    
            }
        }    
    }
    ns_py[0] = ns;
    nd_py[0] = nd;
    n0_py[0] = n0;
    val_py[0] = val;

    '''

    weave.inline(code,['N','M','ns_py','nd_py','n0_py','val_py','weltlinien_neu'])
    
    if val_py !=0:
        print val_py
        exit(1)
    return ns_py, nd_py, n0_py


def autocorr(x):
    result = np.correlate(x, x, mode = 'full')
    maxcorr = np.argmax(result)
    #print 'maximum = ', result[maxcorr]
    result = result / result[maxcorr]     # <=== normalization

    return result[result.size/2:]


t0 = time.time()
# Parameter
anzTeilchen = 16
anzSpinup = 8
anzZeitschritte = 200
termination = anzTeilchen*anzZeitschritte
anzMarkovZeit = 10000

Jz = 1.
Jx = 1.
beta = 10.
print(np.tanh(beta/anzZeitschritte))
# um tatsächlich richte Anzahl der Zeitschritte zu machen
#anzZeitschritte += 1

gs = np.exp(-Jz*beta/anzZeitschritte) * np.cosh(Jx*beta/anzZeitschritte)
gd = np.exp(-Jz*beta/anzZeitschritte) * np.sinh(Jx*beta/anzZeitschritte)
gs_ratio = Jx/anzZeitschritte * np.tanh(beta*Jx/anzZeitschritte) - Jz/anzZeitschritte #gs' / gs
gd_ratio = Jx/anzZeitschritte / np.tanh(beta*Jx/anzZeitschritte) - Jz/anzZeitschritte #gd' / gd

meanNs = np.array([0]*(anzMarkovZeit-1))
meanNd = np.array([0]*(anzMarkovZeit-1))
meanN0 = np.array([0]*(anzMarkovZeit-1))
energy = np.array([0.]*(anzMarkovZeit-1))

# Start
weltlinien = Startconf(anzTeilchen,anzSpinup,anzZeitschritte)
#print(weltlinien)
#print(gewichter(weltlinien))
#print((-ns*gs_ratio - nd*gd_ratio)/anzTeilchen - Jz/4)

## heat up array
#for m in xrange(20):
#    
#    # Loop finden
#    x,y,l,k = loop1(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien)
#
#    # Spinflip
#    for k in xrange(len(x)):
#        weltlinien[int(x[k]),int(y[k])] = weltlinien[int(x[k]),int(y[k])] ^ True

# run simulation
for n in xrange(anzMarkovZeit-1):
    print n    
    
    # Loop finden
    x,y,l,k,update = loop2(anzZeitschritte, anzTeilchen, anzSpinup, termination, weltlinien, Jz, beta, gs, gd)

    # Spinflip
    if update and 0.5 > random.random():
        for k in xrange(len(x)):
            weltlinien[int(x[k]),int(y[k])] = weltlinien[int(x[k]),int(y[k])] ^ True
        
    # Gewichter der Weltlinienkonfiguration berechnen   
    ns, nd, n0 = gewichter(weltlinien)
    
    test_energy = -ns*gs_ratio - nd*gd_ratio
    test_energy /= anzTeilchen
        
    meanNs[n] = int(ns)
    meanNd[n] = int(nd)
    meanN0[n] = int(n0)
    energy[n] = float(test_energy - Jz/4)
        

mean_ns = np.mean(meanNs[anzMarkovZeit/2:anzMarkovZeit])
mean_nd = np.mean(meanNd[anzMarkovZeit/2:anzMarkovZeit])
std_ns = np.std(meanNs[anzMarkovZeit/2:anzMarkovZeit])
std_nd = np.std(meanNd[anzMarkovZeit/2:anzMarkovZeit])

mean_E = -mean_ns*gs_ratio - mean_nd*gd_ratio
mean_E /= anzTeilchen
mean_E -= Jz/4

std_E = std_ns*gs_ratio + std_nd*gd_ratio
std_E /= anzTeilchen

print('E_mean', mean_E)
#print(np.mean(energy[1000::]) - Jz/4)
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
plt.title('Energy = '+str(mean_E)+' +- '+str(std_E))

figPlaketten = plt.figure()
plt.plot(meanN0, '-k', label='S1')
plt.plot(meanNs, '-b', label='S2')
plt.plot(meanNd, '-r', label='S3')
plt.ylabel('Anz Plaketten')
plt.xlabel('Markov Time')
plt.legend()