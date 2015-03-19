# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:51:20 2015

@author: ranftl_s
"""

import numpy


def nearest_neighbours(N,M):
    
    NN = numpy.array([[0,0]]*N*M)
    
    for k in xrange(N*M):
        # oberer Nachbar
        if k < M:  NN[k][0] = (N-1)*M + k
        else: NN[k][0] = k - M
        
        # rechter Nachbar
        if (k+1) % M == 0: NN[k][1] = k - M + 1
        else: NN[k][1] = k + 1
    
    return NN
    
def basis(front0, n1, n2):   
# example: n1 = 8, n2 = 3, k = 0
# front = 00001000
# back =  00000011
# vec =   00001011

    global spin_up, spin_down, spin
    
    for k in xrange(0, n1-n2):
        
        front = 2**(n2+k)
        back = 2**(n2-1)-1
        
        vec = front0 + front + back
        
        vec1 = vec>>4 #füe 2x2 matrix
        spin = numpy.hstack((spin, vec))
        spin_up = numpy.hstack((spin_up, vec1))
        spin_down = numpy.hstack((spin_down, vec - (vec1<<4)))
        
        if n2 > 1:# and not k == 0:
            front1 = front0 + front
            basis(front1, n2+k, n2-1)

def hamilton_diag(spin_up, spin_down, spin, NN, N1):
# N1 = N*M
    
    # dictionary um die bisecs zu den k effizient zu speichern
    flip_list={}

    len_basis = len(spin_up)
    diag = numpy.array([0]*len_basis, dtype=float)
    for k in xrange(len_basis):
        spin_up0 = [int(x) for x in bin(spin_up[k])[2:]]
        spin_up0 = numpy.array([0]*(N1-len(spin_up0)) + spin_up0, dtype=float) 

        spin_down0 = [int(x) for x in bin(spin_down[k])[2:]]
        spin_down0 = numpy.array([0]*(N1-len(spin_down0)) + spin_down0, dtype=float)
        
        spin_total = spin_up0 - spin_down0
        
        # erstelle temporäres array tzr speicherung der bisecs
        flip_tmp=numpy.array([])
        
        for l in xrange(N1):
            if not spin_total[l] == 0:
                diag[k] += ( spin_total[NN[l][0]] + spin_total[NN[l][1]]) * (spin_total[l]) * 0.25
            
            # Wenn die Spins von k und über k unterschiedlich
            if spin_total[NN[l][0]]*spin_total[l] == -1:
                # erstelle eine Maske mit 1en an den Stellen der zu Tauschenden Spins
                mask = (1<<l) | (1<<NN[l][0])
                # und Tausche mit dieser Maske die Spins
                spin_up_neu = spin_up[k] ^ mask
                spin_down_neu = spin_down[k] ^ mask
                spin_neu = (spin_up_neu << 4)
                spin_neu ^= spin_down_neu
                # suche die soeben erstellte Konfiguration in dem Vector aller Konfigurationen
                bisec = bisection(spin,spin_neu)
                # und speichere den Index in einem Vector
                flip_tmp = numpy.hstack((flip_tmp,bisec))
                #print('k:',k,'tmp:',flip_tmp)
                
            # Wenn die Spins von k und neben k unterschiedlich    
            if spin_total[NN[l][1]]*spin_total[l] == -1:
                # erstelle eine Maske mit 1en an den Stellen der zu Tauschenden Spins
                mask = (1<<l) | (1<<NN[l][1])
                 # und Tausche mit dieser Maske die Spins                
                spin_up_neu = spin_up[k] ^ mask
                spin_down_neu = spin_down[k] ^ mask            
                spin_neu = (spin_up_neu << 4)
                spin_neu ^= spin_down_neu
                # suche die soeben erstellte Konfiguration in dem Vector aller Konfigurationen
                bisec = bisection(spin,spin_neu)
                # speichere den Index in einem Vector
                flip_tmp = numpy.hstack((flip_tmp,bisec))

        # Wenn ein Eintrag vorhanden: erstelle einen Dichtionary Eintrag k welcher das Array der bisecs referenziert        
        if flip_tmp.size : flip_list[k] = flip_tmp

    return(diag,flip_list)

def bisection(spin, tofind):
    count = 1
    break_while = 2
    index = (len(spin)>>1)

    while break_while:
        if tofind == spin[index]:
            return(index)
        elif tofind < spin[index]:
            count += 1
            index_shift = len(spin)>>(count)
            if index_shift == 0:
                break_while -= 1
                index -= 1
            else:
                index -= index_shift
        else: 
            count += 1
            index_shift = len(spin)>>(count)
            if index_shift == 0:
                break_while -= 1
                index += 1
            else:
                index += index_shift
            
    return(None)
    

N = 2 # matrix zeile
M = 2# matrix spalte
N2 = 4 #total number of particles with spin up

spin = numpy.array([2**N2-1], dtype=int)
spin_up = numpy.array([0], dtype=int)
spin_down = numpy.array([2**N2-1], dtype=int)

basis(0, 2*N*M, N2)

NN = nearest_neighbours(N,M)

diag,flip = hamilton_diag(spin_up, spin_down, spin, NN, N*M)

print(diag)
print('\n\n')
print(flip) #nun ein dictionary, Zugriff mit flip[k] = numpy.array

