#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
get_ipython().run_line_magic('matplotlib', 'inline')
mysignal = []
mysignal = np.random.randint(2, size=46)
print(mysignal)
print(" ")
for i in range (0,46) :
    if mysignal[i] == 0:
        mysignal[i] = -1
print(mysignal)
print(" ")
Tb = 0.5
A = 5
Eb = (A**2)*Tb
sqEb = np.sqrt(Eb)
base = [np.complex(sqEb,sqEb), np.complex(-sqEb,sqEb), np.complex(-sqEb,-sqEb), np.complex(sqEb,-sqEb)]
x = [x.real for x in base] 
y = [y.imag for y in base] 
plt.plot(x, y, "or")
plt.title('The basic symbols of the constellation QPSK (π/4) Gray coding')
plt.xlabel('In-phase component')
plt.ylabel('Quadrature component')
plt.grid(True)
plt.show()
constellation = []
for i in range (0,46,2) :
    if (mysignal[i] == -1) and (mysignal[i+1] == -1) :
        constellation.append(np.complex(-sqEb,-sqEb))
    if (mysignal[i] == -1) and (mysignal[i+1] == 1) :
        constellation.append(np.complex(-sqEb,sqEb))
    if (mysignal[i] == 1) and (mysignal[i+1] == 1) :
        constellation.append(np.complex(sqEb,sqEb))
    if (mysignal[i] == 1) and (mysignal[i+1] == -1) :
        constellation.append(np.complex(sqEb,-sqEb))
w = [x.real for x in constellation] 
k = [y.imag for y in constellation] 
plt.plot(w, k, "*b")
plt.title('The noiseless constellation of my binary signal QPSK (π/4) Gray coding')
plt.xlabel('In-phase component')
plt.ylabel('Quadrature component')
plt.grid(True)
plt.show()
No1 = Eb / (10**(5/10))
No2 = Eb / (10**(15/10))
X1 = np.random.normal(0, np.sqrt(No1), 46)
X2 = np.random.normal(0, np.sqrt(No2), 46)
constellation1 = []
constellation2 = []
for i in range (0,46,2) :
    print("@@@@@@@@@")
    Esymbol11 = np.sqrt(((A*mysignal[i] + X1[i])**2)*Tb)
    Esymbol12 = np.sqrt(((A*mysignal[i+1] + X1[i+1])**2)*Tb)
    if ((A*mysignal[i] + X1[i]) > 0) and ((A*mysignal[i+1] + X1[i+1]) > 0):
        constellation1.append(np.complex(Esymbol11, Esymbol12))
        print("11")
    if ((A*mysignal[i] + X1[i]) > 0) and ((A*mysignal[i+1] + X1[i+1]) < 0):
        constellation1.append(np.complex(Esymbol11, -Esymbol12))
        print("10")
    if ((A*mysignal[i] + X1[i]) < 0) and ((A*mysignal[i+1] + X1[i+1]) > 0):
        constellation1.append(np.complex(-Esymbol11, Esymbol12))
        print("01")
    if ((A*mysignal[i] + X1[i]) < 0) and ((A*mysignal[i+1] + X1[i+1]) < 0):
        constellation1.append(np.complex(-Esymbol11, -Esymbol12))
        print("00")
    Esymbol21 = np.sqrt(((A*mysignal[i] + X2[i])**2)*Tb)
    Esymbol22 = np.sqrt(((A*mysignal[i+1] + X2[i+1])**2)*Tb)
    if ((A*mysignal[i] + X2[i]) > 0) and ((A*mysignal[i+1] + X2[i+1]) > 0):
        constellation2.append(np.complex(Esymbol21, Esymbol22))
        print("11")
    if ((A*mysignal[i] + X2[i]) > 0) and ((A*mysignal[i+1] + X2[i+1]) < 0):
        constellation2.append(np.complex(Esymbol21, -Esymbol22))
        print("10")
    if ((A*mysignal[i] + X2[i]) < 0) and ((A*mysignal[i+1] + X2[i+1]) > 0):
        constellation2.append(np.complex(-Esymbol21, Esymbol22))
        print("01")
    if ((A*mysignal[i] + X2[i]) < 0) and ((A*mysignal[i+1] + X2[i+1]) < 0):
        constellation2.append(np.complex(-Esymbol21, -Esymbol22))
        print("00")
print("$$$$$$$$$$$$$$$$$$")
print(constellation1)
print("$$$$$$$$$$$$$$$$$$")
print(constellation2)
print("$$$$$$$$$$$$$$$$$$")
w = [w.real for w in constellation1]
k = [k.imag for k in constellation1]
plt.plot(w, k , "*b")
x = [x.real for x in base] 
y = [y.imag for y in base] 
plt.plot(x, y, "or")
plt.title('Constellation diagram of my binary signal QPSK, (π/4) Gray coding, 2-D AWGN and Eb / No equal 5dB')
plt.xlabel('In-phase component')
plt.ylabel('Quadrature component')
plt.grid(True)
plt.show()
w = [w.real for w in constellation2]
k = [k.imag for k in constellation2]
plt.plot(w, k , "*b")
x = [x.real for x in base] 
y = [y.imag for y in base] 
plt.plot(x, y, "or")
plt.title('Constellation diagram of my binary signal QPSK, (π/4) Gray coding, 2-D AWGN and Eb / No equal 15dB')
plt.xlabel('In-phase component')
plt.ylabel('Quadrature component')
plt.grid(True)
plt.show()


bitflow = []
bitflow = np.random.randint(2, size=100000)
for i in range (0,100000) :
    if bitflow[i] == 0:
        bitflow[i] = -1
BER = []
for i in range (0,16):
    constellation = []
    estimate = []
    errors = 0
    No = Eb / (10**(i/10))
    print("The noise is ", No)
    X = np.random.normal(0, np.sqrt(No), 100000)
    for k in range (0,100000,2) :
        Esymbol1 = np.sqrt(((A*bitflow[k] + X[k])**2)*Tb)
        Esymbol2 = np.sqrt(((A*bitflow[k+1] + X[k+1])**2)*Tb)
        if ((A*bitflow[k] + X[k]) > 0) and ((A*bitflow[k+1] + X[k+1]) > 0):
            constellation.append(np.complex(Esymbol1, Esymbol2))
            estimate.append(1)
            estimate.append(1)
        if ((A*bitflow[k] + X[k]) > 0) and ((A*bitflow[k+1] + X[k+1]) < 0):
            constellation.append(np.complex(Esymbol1, -Esymbol2))
            estimate.append(1)
            estimate.append(-1)
        if ((A*bitflow[k] + X[k]) < 0) and ((A*bitflow[k+1] + X[k+1]) > 0):
            constellation.append(np.complex(-Esymbol1, Esymbol2))
            estimate.append(-1)
            estimate.append(1)
        if ((A*bitflow[k] + X[k]) < 0) and ((A*bitflow[k+1] + X[k+1]) < 0):
            constellation.append(np.complex(-Esymbol1, -Esymbol2))
            estimate.append(-1)
            estimate.append(-1)
        if (bitflow[k] != estimate[k]) and (bitflow[k+1] != estimate[k+1]) :
            errors = errors + 2
        if (bitflow[k] == estimate[k]) and (bitflow[k+1] != estimate[k+1]) :
            errors = errors + 1
        if (bitflow[k] != estimate[k]) and (bitflow[k+1] == estimate[k+1]) :
            errors = errors + 1
    print("The errors are ", errors)
    BER.append(errors/100000)
    w = [w.real for w in constellation] 
    k = [k.imag for k in constellation] 
    plt.plot(w, k, "*b")
    x = [x.real for x in base] 
    y = [y.imag for y in base] 
    plt.plot(x, y, "or")
    print('Constellation diagram of my binary signal with complex AWGN and Eb / No equal ', i, 'dB.' )
    plt.grid(True)
    plt.show()
    print("%%%%%%%%%%%")
print(BER)
noise = np.arange (0, 16, 1)
plt.plot(noise, BER)
plt.title("BER for my flow of 100000 bits -logarithmic scale")
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.yscale('log',base=10) 
plt.show()
noise = np.arange (0, 16, 1)
plt.plot(noise, BER)
plt.title("BER for my flow of 10000 bits - linear scale")
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.show()


# In[ ]:




