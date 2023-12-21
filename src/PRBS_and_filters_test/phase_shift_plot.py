# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:10:01 2023

@author: thoma
"""

import matplotlib.pyplot as plt

alpha1 = 0.4

alpha2 = 0.6

time = [i/(85000 *200)  for i in range(401)]
voltage = [0]

V_dc = 40

for i in range(400):
    if i%200  < 100 : 
        if i%100   > 100 * alpha1:
            voltage.append(0)
        else:
            voltage.append(V_dc)
    
    if i%200 >= 100 : 
        if i%100   > 100 * alpha2:
            voltage.append(0)
        else:
            voltage.append(-V_dc)
            
plt.plot(time, voltage)

plt.show()