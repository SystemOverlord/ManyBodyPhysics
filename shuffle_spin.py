# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 20:35:48 2015

@author: lukashoermann
"""

import numpy
   
def func(front0, n1, n2):   
    global output
    
    for k in range(1, n1-n2+1):
        
        front = numpy.array([0]*(n1-n2-k) + [1], dtype=bool)
        back = numpy.array([0]*k + [1]*(n2-1), dtype=bool)
        
        vec = numpy.concatenate((front0, front, back), axis=1)
        output = numpy.vstack((output, vec))
        
        if n2 > 1 and not k == 0:
            front1 = numpy.concatenate((front0, front), axis=1)
            func(front1, n2-1+k, n2-1)

N1 = 6
N2 = 3

vec0 = numpy.array([], dtype=bool)
output = numpy.array([0]*(N1-N2) + [1]*(N2), dtype=bool)

func(vec0, N1, N2)

print(output)







