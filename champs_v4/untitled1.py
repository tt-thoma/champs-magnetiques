# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:03:46 2024

@author: coulo
"""
import numpy as np
import random as rd

N = 9
L = 100

principale = [rd.randint(0, rd.randint(0, 1)) for i in range(L)]
print(principale)

l_s = [[rd.randint(0, rd.randint(1, 2)) for ii in range(L)] for i in range(N)]

n = 0
et = []
for n in range(len(principale)):
    i = 0
    while principale[n] == 0:
        if i < len(l_s):
            if l_s[i][n] == 0:
                principale[n] = 1
                l_s[i][n] = -10
                et.append(l_s[i])
                l_s.pop(i)

                et.append(l_s[i])
            else:
                i += 1
        else:
            print("error")
            break
print(principale)
print(et)
ett = 0
for l in et:
    for i in range(len(l)):
        if l[i] == -10:
            ett += 1
print(ett / N)
