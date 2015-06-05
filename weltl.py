# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:14:26 2015

@author: olaf
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import traceback
    
    
def Gitterplot(gitter):
    anzZeitschritte = np.shape(gitter)[0]
    anzTeilchen = np.shape(gitter)[1]
    fig1 = plt.figure()
    fig1.show()
    for i in xrange(anzTeilchen):    
        for t in xrange(anzZeitschritte):    
            if gitter[t][i]:
                plt.plot(i,t,'ro')    
    plt.xlim([-0.5,anzTeilchen-0.5])
    plt.ylim([-0.5,anzZeitschritte-0.5])

def Startconf(anzTeilchen,anzSpinUp,anzZeitschritte):
    conf=[]
    for i in xrange(anzTeilchen):
        conf.append(0)
    
    i = 0
    while i < anzSpinUp:
        tmp = random.randint(0, anzTeilchen-1)
        #print('tmp:',tmp)
        if conf[tmp] == 0:
            conf[tmp] = 1
            i = i+1
    #print(conf)
    gitter = []
    #print(np.shape(gitter))
    for t in xrange(anzZeitschritte):
       gitter.append(conf)
    return gitter

class loop:

    gitter=[]       
    imax_plaquetten=0
    tmax_plaquetten=0
    def __init__(self,Gitter):
        self.gitter = Gitter
        self.tmax_plaquetten = np.shape(gitter)[0]        
        self.imax_plaquetten = np.shape(gitter)[1]
        
    def getFeld(self,T,I):
        Ttest = T>=self.tmax_plaquetten-1 or T<0
        Itest = I>=self.imax_plaquetten-1 or I<0
        if Ttest or Itest:
            traceback.print_stack()
            print 'Index is net in Range!:', T,'/',self.tmax_plaquetten-2,I,'/',self.imax_plaquetten-2
            return
        
        erg = []
        erg.append(gitter[T][I])
        erg.append(gitter[T][I+1])
        erg.append(gitter[T+1][I])
        erg.append(gitter[T+1][I+1])
        return erg
            
    def isFeldActive(self,T,I):
        return T%2==I%2
    
    def getDirection(self,T,I):
        feld = self.getFeld(T,I)
        erg = []
        erg.append(feld[0]^0)
        erg.append(feld[1]^0)
        erg.append(feld[2]^1)
        erg.append(feld[3]^1)
        return erg
    
    def randFeld(self):
        while bool(1):    
            i = random.randint(0, self.imax_plaquetten -2)
            t = random.randint(0, self.tmax_plaquetten -2)
            if self.isFeldActive(t,i):
                return t,i
                
    def betretbaresFeld(self,t,i):
        richtungen = self.getDirection(t,i)
        summe = 0
        for n in xrange(4):
            summe += richtungen[n]
        return summe == 2
                
    def calcLoob(self):
        ft,fi = self.randFeld()
        loop = [[ft,fi]]      
        changespin = [[-1,-1]]
        while 1:
            richtungen = self.getDirection(ft,fi)

            while 1:            
                choose = random.randint(0,3)            
                if richtungen[choose]:
                    break

            offset_i = choose%2 if choose%2 else -1
            offset_t = choose/2 if choose/2 else -1
            ft_new = ft + offset_t
            fi_new = fi + offset_i            
            ft_new = ft_new if ft_new <self.tmax_plaquetten-1 else ft_new-self.tmax_plaquetten+1
            fi_new = fi_new if fi_new <self.imax_plaquetten-1 else fi_new-self.imax_plaquetten+1
            ft_new = ft_new if ft_new >=0 else ft_new+self.tmax_plaquetten-1
            fi_new = fi_new if fi_new >=0 else fi_new+self.imax_plaquetten-1
            
            if not self.betretbaresFeld(ft_new,fi_new):
                continue

            changespin.append([ft+choose/2,fi+choose%2])            
            
            ft = ft_new
            fi = fi_new
                     
            for n in xrange(1,np.shape(loop)[0]):
                if loop[n][0] == ft and loop[n][1] == fi:
                    loop.append([ft,fi])
                    changespin = changespin[n+1:np.shape(changespin)[0]]
                    loop = loop[n:np.shape(loop)[0]]
                    return loop, changespin
                    
            loop.append([ft,fi])
                    
        
if __name__ == '__main__':    

    anzTeilchen = 5     # muss durch 2 teilbar sein
    anzSpinUp = 2
    anzZeitschritte = 10  # muss durch 2 teilbar sein
    
    gitter = Startconf(anzTeilchen,anzSpinUp,anzZeitschritte)
    print np.array(gitter),'\n'
    
    temp = loop(gitter)
    print temp.getFeld(0,0), '\n'
    print temp.getDirection(0,0)
    
    loop, changespin = temp.calcLoob()
    loop = np.transpose(loop)
    changespin = np.transpose(changespin)
    
#    indizesYX = FeldindizesYX(gitter)
#    
    print '\nloop:\n',loop,'\nspins:\n',changespin
    
    Gitterplot(gitter)
    plt.plot(changespin[1],changespin[0],'bo')
    plt.plot(loop[1]+0.5,loop[0]+0.5,'go')
    for n in xrange(np.shape(loop)[1]-1):
        t0 = loop[0][n]+0.5
        t1 = loop[0][n+1]+0.5
        i0 = loop[1][n]+0.5
        i1 = loop[1][n+1]+0.5        
        plt.arrow(i0,t0,i1,t1,length_includes_head=True,head_width=0.1, head_length=0.2)
        print 'i:', i0,i1,'t:',t0,t1
    
    #
    #plt.plot()
#    loop, plaquettenBesucht = Loopalg(gitter)    
#    for i in xrange(np.shape(loop)[0]):
#        plt.plot(loop[i][1],loop[i][0],'bo')
#        
#    plaquettenPlot = np.array([[-1,-1]])
#    vx=[]
#    vy=[]
#    for i in xrange(np.shape(plaquettenBesucht)[0]):
#        tx = indizesYX[plaquettenBesucht[i,0],plaquettenBesucht[i,1]][0,1]+0.5
#        ty = indizesYX[plaquettenBesucht[i,0],plaquettenBesucht[i,1]][0,0]+0.5
#        vx.append(tx)
#        vy.append(ty)
#    #plt.figure()
#    plt.plot(vx,vy,'go')
#    
#    
#    #print gitter[(indizesYX[0][0][0][0]),(indizesYX[0][0][0][1])]
#    #print indizesYX[2,2]
    

    
    