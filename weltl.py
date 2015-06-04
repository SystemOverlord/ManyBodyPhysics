# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pprint
    
def FeldindizesYX(gitter):
    xmax = np.shape(gitter)[0]-1
    ymax = np.shape(gitter)[1]-1
    indizesYX = np.array([[[[0]*2]*4]*xmax]*ymax)
    
    for y in xrange(ymax):
        for x in xrange(xmax):
            #print('x:',x,'/',xmax,'y:',y,'/',ymax,'shape:',np.shape(indizesYX))
            indizesYX[y][x][0][0] = x     # unten links X
            indizesYX[y][x][0][1] = y     # unten links Y       
            indizesYX[y][x][1][0] = x+1   # unten rechts X
            indizesYX[y][x][1][1] = y     # unten rechts Y            
            indizesYX[y][x][2][0] = x     # oben links X
            indizesYX[y][x][2][1] = y+1   # oben links Y
            indizesYX[y][x][3][0] = x+1   # oben rechts X
            indizesYX[y][x][3][1] = y+1   # oben rechts Y
            

#            if x == 0 and y == 0:
#                indizesBereinigt = [indizesYX[y][x]]
#            if (y%2 == x%2) and not (x==0 and y==0): 
#                #print np.shape(indizesYX[y][x]),np.shape(indizesBereinigt)
#                indizesBereinigt = np.concatenate((indizesBereinigt,[indizesYX[y][x]]))
    return indizesYX#, indizesBereinigt
    
def Gitterplot(gitter):
    anzZeitschritte = np.shape(gitter)[0]
    anzTeilchen = np.shape(gitter)[1]
    fig1 = plt.figure()
    fig1.show()
    for t in xrange(anzZeitschritte):
        for i in xrange(anzTeilchen):
            if gitter[t][i]:
                plt.plot(i,t,'ro')    
    plt.xlim([-0.5,anzTeilchen-0.5])
    plt.ylim([-0.5,anzZeitschritte-0.5])

def Startconf(anzTeilchen,anzSpinUp,anzZeitschritte):
    conf = np.array([int(0)]*anzTeilchen)
    i = 0
    while i < anzSpinUp:
        tmp = random.randint(0, anzTeilchen-1)
        #print('tmp:',tmp)
        if conf[tmp] == 0:
            conf[tmp] = 1
            i = i+1
    #print(conf)
    gitter = np.array([[-1]*anzTeilchen]*anzZeitschritte)
    #print(np.shape(gitter))
    for t in xrange(anzZeitschritte):
        for i in xrange(anzTeilchen):
            gitter[t][i] = conf[i];
    return gitter
         
         
#===== 1 richtung plaquette    0 raus aus plaquette
def Feldrichtung(feld,gitter):
    richtung = np.array([[1,1],[0,0]])
    #Gitterplot( gitter[feld[0,0]:feld[3,0]+1,feld[0,1]:feld[3,1]+1] )
    #print( gitter[feld[0,0]:feld[3,0]+1,feld[0,1]:feld[3,1]+1])
    #print(richtung)
    richtung ^= gitter[feld[0,0]:feld[3,0]+1,feld[0,1]:feld[3,1]+1]
    #print(richtung)
    return richtung

def Loopalg(gitter):
    indizesYX = FeldindizesYX(gitter)
    
    while bool(1):           
        x = random.randint(0, np.shape(indizesYX)[1]-1)
        y = random.randint(0, np.shape(indizesYX)[0]-1)
        if (y%2 == x%2) and not (x==0 and y==0):                     
            break
    
    #print('richtung')
    
    plaquettenBesucht = np.array([[-1,-1]])
    spinsToChange = np.array([[-1,-1]])

    while bool(1):     
        #print('x:',x,'y:',y,'shape:',np.shape(indizesYX))
        #print('idizes:',indizesYX[y][x])
        feld = indizesYX[y][x]
        richtung = Feldrichtung(feld,gitter)    
        while bool(1):    
            rx = random.randint(0, 1)
            ry = random.randint(0, 1)              
            if richtung[rx][ry]:
                spinsToChange = np.vstack((spinsToChange,feld[rx+2*ry]))
                rx = -1 if rx == 0 else rx 
                ry = -1 if ry == 0 else ry
                y += ry
                x += rx
                
                y = y + np.shape(indizesYX)[0] if y<0 else y
                x = x + np.shape(indizesYX)[1] if x<0 else x
                y = y - np.shape(indizesYX)[0] if y>=np.shape(indizesYX)[0] else y
                x = x - np.shape(indizesYX)[1] if x>=np.shape(indizesYX)[1] else x
                print y,x, np.shape(indizesYX)
                break   
        #print plaquettenBesucht,'=',np.shape(plaquettenBesucht),'\n'
        for k in xrange(1,np.shape(plaquettenBesucht)[0]):
            if plaquettenBesucht[k][0] == y and plaquettenBesucht[k][1] == x:
                #print plaquettenBesucht[k][0],plaquettenBesucht[k][1]
                spinsToChange = spinsToChange[k:np.shape(spinsToChange)[0]]
                plaquettenBesucht = plaquettenBesucht[k:np.shape(plaquettenBesucht)[0]]
                plaquettenBesucht = np.vstack((plaquettenBesucht,np.array([y,x])))
                return spinsToChange, plaquettenBesucht
                
            
        plaquettenBesucht = np.vstack((plaquettenBesucht,np.array([y,x])))
    
    #print(richtung,rx,ry,richtung[rx][ry])    

            
    
    # XOR
    # 0 0
    # 1 1  
def loop(gitter):
    
        
    


if __name__ == '__main__':    

    anzTeilchen = 6      # muss durch 2 teilbar sein
    anzSpinUp = 3
    anzZeitschritte = 10  # muss durch 2 teilbar sein
    
    gitter = Startconf(anzTeilchen,anzSpinUp,anzZeitschritte)
    
    indizesYX = FeldindizesYX(gitter)
    
    Gitterplot(gitter)
    loop, plaquettenBesucht = Loopalg(gitter)    
    for i in xrange(np.shape(loop)[0]):
        plt.plot(loop[i][1],loop[i][0],'bo')
        
    plaquettenPlot = np.array([[-1,-1]])
    vx=[]
    vy=[]
    for i in xrange(np.shape(plaquettenBesucht)[0]):
        tx = indizesYX[plaquettenBesucht[i,0],plaquettenBesucht[i,1]][0,1]+0.5
        ty = indizesYX[plaquettenBesucht[i,0],plaquettenBesucht[i,1]][0,0]+0.5
        vx.append(tx)
        vy.append(ty)
    #plt.figure()
    plt.plot(vx,vy,'go')
    
    
    #print gitter[(indizesYX[0][0][0][0]),(indizesYX[0][0][0][1])]
    #print indizesYX[2,2]
    

    
    