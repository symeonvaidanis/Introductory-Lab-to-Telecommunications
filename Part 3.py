#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
import cmath 
import math
get_ipython().run_line_magic('matplotlib', 'inline')
mysignal = []
mysignal = np.random.randint(2, size=46)
for i in range (0,46) :
    if mysignal[i] == 0:
        mysignal[i] = -1
Tb = 0.5
A = 5
Eb = (A**2)*Tb
sqEb = np.sqrt(Eb)
base = [-sqEb, sqEb]
for i in range (0,46) :
    m = i*Tb
    n = (i+1)*Tb
    t = np.arange(m, n,0.05)
    plt.plot(t, mysignal[i]*A*signal.square(t, duty=1), 'b')
plt.title('My binary signal B-PAM modulation')
plt.xlabel('time(seconds)')
plt.ylabel('Volt(V)')
plt.grid(True)
plt.show()
plt.plot(sqEb*mysignal, len(mysignal) * [0], '*r')
plt.title('Constellation diagram of my binary signal')
plt.grid(True)
plt.show()
No1 = Eb / (10**(5/10))
No2 = Eb / (10**(15/10))
X1 = np.random.normal(0, np.sqrt(No1/2), 46)
X2 = np.random.normal(0, np.sqrt(No2/2), 46)
Y1 = np.random.normal(0, np.sqrt(No1/2), 46)
Y2 = np.random.normal(0, np.sqrt(No2/2), 46)
mysignal1 = []
mysignal2 = []
for i in range (0,46) :
    mysignal1.append(A*mysignal[i] + X1[i])
    mysignal2.append(A*mysignal[i] + X2[i])
for i in range (0,46) :
    m = i*Tb
    n = (i+1)*Tb
    t = np.arange(m, n,0.05)
    plt.plot(t, mysignal1[i]*signal.square(t, duty=1), 'r')
plt.title('My binary signal B-PAM modulation with Eb / No equal 5dB')
plt.xlabel('time(seconds)')
plt.ylabel('Volt(V)')
plt.grid(True)
plt.show()
for i in range (0,46) :
    m = i*Tb
    n = (i+1)*Tb
    t = np.arange(m, n,0.05)
    plt.plot(t, mysignal2[i]*signal.square(t, duty=1), 'g')
plt.title('My binary signal B-PAM modulation Eb / No equal 15dB')
plt.xlabel('time(seconds)')
plt.ylabel('Volt(V)')
plt.grid(True)
plt.show()
constellation1 = []
constellation2 = []
for i in range (0,46) :
    if A*mysignal[i] + X1[i] > 0:
        constellation1.append(np.complex(np.sqrt(Tb*((A*mysignal[i] + X1[i])**2)), Y1[i]))
    if A*mysignal[i] + X2[i] > 0:
        constellation2.append(np.complex(np.sqrt(Tb*((A*mysignal[i] + X2[i])**2)), Y2[i]))
    if A*mysignal[i] + X1[i] < 0:
        constellation1.append(np.complex(-np.sqrt(Tb*((A*mysignal[i] + X1[i])**2)), Y1[i]))
    if A*mysignal[i] + X2[i] < 0:
        constellation2.append(np.complex(-np.sqrt(Tb*((A*mysignal[i] + X2[i])**2)), Y2[i]))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(constellation1)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(constellation2)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
plt.plot(base, len(base) * [0], 'or')
#plt.plot(constellation1, len(constellation1) * [0], '*b') 
x = [x.real for x in constellation1] 
y = [y.imag for y in constellation1] 
plt.plot(x, y, "*b")
plt.title('Constellation diagram of my binary signal with complex AWGN and Eb / No equal 5dB')
plt.grid(True)
plt.show()
plt.plot(base, len(base) * [0], 'or')
#plt.plot(constellation2, len(constellation2) * [0], '*b')  
x = [x.real for x in constellation2] 
y = [y.imag for y in constellation2] 
plt.plot(x, y, "*b")
plt.title('Constellation diagram of my binary signal with complex AWGN and Eb / No equal 15dB')
plt.grid(True)
plt.show()

bitflow = []
bitflow = np.random.randint(2, size=10000)
for i in range (0,10000) :
    if bitflow[i] == 0 :
        bitflow[i] = -1
BER = []
for i in range (0,16) :
    constellation = []
    estimate = []
    errors = 0
    No = Eb / (10**(i/10))
    print(No)
    X = np.random.normal(0, np.sqrt(No/2), 10000)
    Y = np.random.normal(0, np.sqrt(No/2), 10000)
    check = []
    for k in range (0,10000) :
        check.append(A*bitflow[k] + X[k])
        if A*bitflow[k] + X[k] > 0:
            constellation.append(np.complex(np.sqrt(Tb*((A*bitflow[k] + X[k])**2)), Y[k]))
        if A*bitflow[k] + X[k] < 0:
            constellation.append(np.complex(-np.sqrt(Tb*((A*bitflow[k] + X[k])**2)), Y[k]))
        if constellation[k].real >= 0 :
            estimate.append(1)
        else :
            estimate.append(-1)
        if estimate[k] != bitflow[k] :
            errors = errors + 1
    for j in range (0,1000) :
        m = j*Tb
        n = (j+1)*Tb
        t = np.arange(m, n,0.05)
        plt.plot(t, check[j]*signal.square(t, duty=1), 'g')
    print('My binary signal B-PAM modulation Eb / No equal ', i, 'dB.')
    plt.xlabel('time(seconds)')
    plt.ylabel('Volt(V)')
    plt.grid(True)
    plt.show()
    #plt.plot(constellation1, len(constellation1) * [0], '*b') 
    x = [x.real for x in constellation] 
    y = [y.imag for y in constellation] 
    plt.plot(x, y, "*b")
    plt.plot(base, len(base) * [0], 'or')
    print('Constellation diagram of my binary signal with complex AWGN and Eb / No equal ', i, 'dB.' )
    plt.grid(True)
    plt.show()
    print(errors)
    BER.append(errors/10000)
    print("%%%%%%%%%%%%%%%%%%%%%%")
print(BER)
noise = np.arange (0, 16, 1)
plt.plot(noise, BER)
plt.title("BER for my flow of 10000 bits -logarithmic scale")
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




