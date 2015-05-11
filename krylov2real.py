# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:40:46 2015

@author: ranftl_s
"""

import numpy
from scipy import weave
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def nearest_neighbours(N,M):
    NN = numpy.array([[0,0]]*N*M)
    
    for k in xrange(N*M):
        
        # oberer Nachbar
        if k < M: NN[k][0] = (N-1)*M + k
        else: NN[k][0] = k - M
        
        # rechter Nachbar
        if (k+1) % M == 0: NN[k][1] = k - M + 1
        else: NN[k][1] = k + 1
        
    return NN
    
def basis(front0, n1, n2, N1):
    # example: n1 = 8, n2 = 3, k = 0
    # front = 00001000
    # back = 00000011
    # spin_add = 00001011

    global spin_up, spin_down, spin
    
    for k in xrange(0, n1-n2):
        front = 1<<(n2+k) #2**(n2+k)
        back = ( 1<<(n2-1) ) - 1  #2**(n2-1)-1
        
        spin_add = front0 + front + back
        spin_up_add = spin_add>>N1 #fuer 2x2 matrix
        
        spin = numpy.hstack((spin, spin_add))
        spin_up = numpy.hstack((spin_up, spin_up_add))
        spin_down = numpy.hstack((spin_down, spin_add - (spin_up_add<<N1)))
        
        if n2 > 1:
            front1 = front0 + front
            basis(front1, n2+k, n2-1, N1)
    
def hamilton_diag(spin_up, spin_down, spin, NN, N1):
    # N1 = N*M
    # dictionary um die bisecs zu den k effizient zu speichern
    H = numpy.array([[0,0,0]], dtype=float)
    len_basis = len(spin_up)
    # initialisiere diagonale der hamiltom matrix
    
    for k in xrange(len_basis):
        
        # erstelle temporäres array zur speicherung der bisecs
        # ([m, n, Betrag Matrixelement])
        flip_tmp = numpy.array([[0,0,0]], dtype=int)
        
        # initialisiere Diagonalelement
        diag_neu0 = 0
        
        for l in xrange(N1):
            mask_tot = 1<<(N1-l-1)
            
            #if not spin_total[l] == 0:
            if (spin_up[k] ^ spin_down[k]) & mask_tot:
                mask1 = 1<<(N1-NN[l][0]-1)
                mask2 = 1<<(N1-NN[l][1]-1)

                S1 = bool(spin_up[k] & mask_tot) #spin_up an postion l
                S2 = bool(spin_down[k] & mask_tot) #spin_down an postion l
                
                S1_NN = bool(spin_up[k] & mask1) #oberen NN von l ist spin_up
                S2_NN = bool(spin_up[k] & mask2) #rechter NN von l ist spin_up
                S3_NN = bool(spin_down[k] & mask1) #oberen NN von l ist spin_down
                S4_NN = bool(spin_down[k] & mask2) #rechter NN von l ist spin_down
                
                # berechne Diagonalelement
                if S1:
                    if S1_NN: diag_neu0 += 0.25
                    if S2_NN: diag_neu0 += 0.25
                    if S3_NN: diag_neu0 -= 0.25
                    if S4_NN: diag_neu0 -= 0.25

                if S2:
                    if S3_NN: diag_neu0 += 0.25
                    if S4_NN: diag_neu0 += 0.25
                    if S1_NN: diag_neu0 -= 0.25
                    if S2_NN: diag_neu0 -= 0.25
                
                # Flipoperator mit Spin oberhalb
                # 1. ob überhaupt Spin an der Position des nächsten Nachbarn
                # 2. ob Spins an Position l und an der Position des nächsten Nachbar unterschiedlich sind
                if (spin_up[k] ^ spin_down[k]) & mask1 and (S1_NN & S2 or S3_NN & S1):
                    # erstelle eine Maske mit 1en an den Stellen der zu Tauschenden Spins
                    mask = (1<<(N1-l-1)) | (1<<(N1-NN[l][0]-1))
                    
                    # und Tausche mit dieser Maske die Spins
                    spin_up_neu = spin_up[k] ^ mask
                    spin_down_neu = spin_down[k] ^ mask
                    spin_neu = (spin_up_neu << N1)
                    spin_neu ^= spin_down_neu
                    
                    # suche die soeben erstellte Konfiguration in dem Vector aller Konfigurationen
                    bisec = bisection(spin,spin_neu)

                    # trigger um Matrixelement zu speichern
                    index_neu = 1
                    # such in array der Matrixelemente nach doppeltem Eintrag
                    # bei doppeltem Eintrag addiere Matrixelemente
                    for m in xrange(1,len(flip_tmp)):                     
                        
                        if bisec == flip_tmp[m][1]:
                            flip_tmp[m][2] += 1
                            index_neu = 0
                            break
                    
                    # und speichere den Index in einem Vector
                    if index_neu == 1:
                        flip_tmp = numpy.vstack((flip_tmp, numpy.array([[k,bisec,1]])))
                
                # Flipoperator mit Spin rechts
                # 1. ob überhaupt Spin an der Position des nächsten Nachbarn
                # 2. ob Spins an Position l und an der Position des nächsten Nachbar unterschiedlich sindv
                if (spin_up[k] ^ spin_down[k]) & mask2 and (S2_NN & S2 or S4_NN & S1):
                    # erstelle eine Maske mit 1en an den Stellen der zu Tauschenden Spins
                    mask = (1<<(N1-l-1)) | (1<<(N1-NN[l][1]-1))
                    
                    # und Tausche mit dieser Maske die Spins
                    spin_up_neu = spin_up[k] ^ mask
                    spin_down_neu = spin_down[k] ^ mask
                    spin_neu = (spin_up_neu << N1)
                    spin_neu ^= spin_down_neu
                    
                    # suche die soeben erstellte Konfiguration in dem Vector aller Konfigurationen
                    bisec = bisection(spin,spin_neu)

                    # trigger um Matrixelement zu speichern
                    index_neu = 1
                    # such in array der Matrixelemente nach doppeltem Eintrag
                    # bei doppeltem Eintrag addiere Matrixelemente
                    for m in xrange(1,len(flip_tmp)):                     
                        
                        if bisec == flip_tmp[m][1]:
                            flip_tmp[m][2] += 1
                            index_neu = 0
                            break
                    
                    # und speichere den Index in einem Vector
                    if index_neu == 1:
                        flip_tmp = numpy.vstack((flip_tmp, numpy.array([[k,bisec,1]])))
        
        # speichere Diagonalelement wenn ungleich 0
        if not diag_neu0 == 0:
            diag_neu = numpy.array([k,k, diag_neu0])
            H = numpy.vstack((H, diag_neu))
              
        # Wenn ein Eintrag vorhanden: schpeichere flip_tem in flip_list
        if flip_tmp.size-1: H = numpy.vstack((H, flip_tmp[1::]))
        
    return H[1::] #diag
    
def bisection(spin, tofind):
    count = 1
    break_while = 12
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

def lanczos(H, size_basis, delta, n_max, n_diag):
    
    Hx1 = numpy.array([0.0]*size_basis) # initialisiere Produkt von H mit Vektor x_n
    x0 = numpy.array([0.0]*size_basis) # initialisiere Vektor x_n-1
    x1 = numpy.random.rand(size_basis) # initialisiere Vektor x_n
    x1 /= numpy.sqrt(x1.dot(x1))
    
    epsilon = numpy.array([0.0]*(n_max+1))
    k = numpy.array([0.0]*(n_max+1))
    
    E = [] 
    
    breakvar=0;
    for n in xrange(size_basis):

        if n > n_max: break
        
        if n > 0:
            k[n] = numpy.sqrt(x1.dot(x1))
            
            if k[n] < delta: breakvar=1
            else: x1 /= k[n]
            
        Hx1 = H.dot(x1)
        epsilon[n] = x1.dot(Hx1)
        
        if (((n+1) % n_diag) == 0) or (breakvar == 1):
            
            a = numpy.array(epsilon[0:n])
            b = numpy.array(k[1:n])
            
            E0,sumb = QR(a,b,0.001,100000)
            E += [numpy.array([min(E0), max(E0)])]            
            
            if len(E) > 1:
                if abs(E[-2][0] - E[-1][0]) < 1e-8 and abs(E[-2][1] - E[-1][1]) < 1e-8:
                    breakvar = 1
            
            print(n_max,n,sumb)
        
        if (breakvar==1): break
        
        Hx1 = Hx1 - epsilon[n]*x1 - k[n]*x0
        x0 = x1
        x1 = Hx1
        
    return(x1, epsilon, k, E, a,b,n)
    
    
def lanczos2(H, size_basis, n_max, c_matrix):
    
    Hx1 = numpy.array([0.0]*size_basis) # initialisiere Produkt von H mit Vektor x_n
    x0 = numpy.array([0.0]*size_basis) # initialisiere Vektor x_n-1
    x1 = numpy.random.rand(size_basis) # initialisiere Vektor x_n
    x1 /= numpy.sqrt(x1.dot(x1))
    psi = numpy.array([0.0]*n_max]) # initialisiere Ortsraumvektor
    k = 0    
    
    for n in xrange(n_max):
        
        if n > 0:
            k = numpy.sqrt(x1.dot(x1))
            x1 /= k
            
        Hx1 = H.dot(x1)
        epsilon = x1.dot(Hx1)
                
        Hx1 = Hx1 - epsilon*x1 - k*x0
        x0 = x1
        x1 = Hx1
        
        # Transformiere von Krylovbasis in Eigenbasis
        psi[n] = c_vector.dot(x1)
        
    return(psi)

def QR(a,b,delta,h_max):
    # QR Algorithmus mittels Givens Rotations Matrizen
    # für Tridiagonalmatrix
    # a ... Hauptdiagonale
    # b ... Nebendiagonale

    n = len(a)
    
    code = r'''
    
        for(int h = 0; h < h_max; h++) {
            
            double t = b[0];
            double sum_b = 0;
            double c[n] = { 0 };
            double s[n] = { 0 };            
            
            // erzeugen von R = G * A
            for(int i = 0; i < n-1; i++) {
                
                double r = sqrt(a[i]*a[i] + t*t);
                c[i] = a[i] / r;
                s[i] = t / r;
                
                t = b[i];
                
                a[i] = r;
                b[i] = t*c[i] + a[i+1]*s[i];
                a[i+1] = -t*s[i] + a[i+1]*c[i];
                
                if(i != n-2) {
                    t = b[i+1];
                    b[i+1] = t*c[i];
                }
            }
            
            // multiplikation mit Q = R * (G)T
            for(int j = 0; j < n-1; j++) {

                a[j] = a[j]*c[j] + b[j]*s[j];
                b[j] = a[j+1]*s[j];
                a[j+1] = a[j+1]*c[j];
                
                double add = b[j];
                if(add < 0) {add = -add;}
                sum_b = sum_b + add;
            }
            
            return_val = sum_b;
            
            if(sum_b < delta) {break;}
            
        }
       
    '''

    sum_b = weave.inline(code,['n','a','b','delta','h_max'])
    
    return a,sum_b

    
# Main

N = 3 # matrix zeile
M = 3 # matrix spalte
N2 = 5 #total number of particles with spin up

spin = numpy.array([2**N2-1], dtype=int)
spin_up = numpy.array([0], dtype=int)
spin_down = numpy.array([2**N2-1], dtype=int)

# berechne alle Basiszustände
print('basis')
basis(0, 2*N*M, N2, N*M)

# berechne Indizes der Nächste Nachbarn
print('nearest neighbors')
NN = nearest_neighbours(N,M)

# berechne Matrix des Heisenberg Hamilton Operators
print('hamiltonian')
H = hamilton_diag(spin_up, spin_down, spin, NN, N*M)

# transformiere H in eine sparse Matrix
print('csr_matrix')
H = csr_matrix(( H[:,2], (H[:,0], H[:,1]) ), shape=(len(spin), len(spin)))

# lanczos
print('lanczos')
x, epsilon, k, E, a, b, n_runs = lanczos(H, len(spin), 1e-2, 10000, 2)

# Eigenvektoren
ind = 0 #index des (größten) Eigenwerts
c_vector = spsolve(H - E[ind]*numpy.eye(len(spin)), numpy.array([0.0]*len(spin)))

# lanczos2
print('lanczos2')
psi = lanczos2(H, len(spin), n_runs, c_vector)



# plot
print('plotten')

fig2 = plt.figure()
fig2.show()
plt.xlim(-1,len(E))

for i in range(len(E)):
    plt.plot(i,E[i][0],'or')
    plt.plot(i,E[i][1],'or')
    print(E[i][0], E[i][1])

# zu Testzwecken fertiger Lanczos
print eigsh(H)[0]
