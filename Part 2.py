#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
get_ipython().run_line_magic('matplotlib', 'inline')
def gray(n) : 
    z = n ^ n >> 1
    w = bin(z).replace("0b", "") 
    return w
fm = 5000
fs1 = 20 * fm 
Ts1 = 1 / (fs1)
AM = 5
t1 = np.arange(0,0.00041,Ts1)   # start,stop,step
y1 = np.cos(2*np.pi*fm*t1)*np.cos(2*np.pi*(AM + 2)*fm*t1)
plt.stem(t1, y1)
plt.title('Figure of y(t) with sampling frequency fs1 = 20fm')
plt.xlabel('time(s)')
plt.ylabel('y1(V)')
plt.grid(True)
plt.show()
R = 5
L = 2 ** R
y1max = 1
y1min = np.cos(2*np.pi*fm*Ts1)*np.cos(2*np.pi*(AM + 2)*fm*Ts1)
D = (y1max - y1min) / L
q = []
quantizer = []
quantizergray = []
yvalues = []
receiver = []
print('Step value of Mid-Riser Quantizer : ',D)
i = 0
j = 0
mean1 = 0
mean2= 0
meanq1 = 0
meanq2 = 0
while True:
    x = np.cos(2*np.pi*fm*j)*np.cos(2*np.pi*(AM + 2)*fm*j)
    yvalues.append(x)
    quantizer.append(np.floor(x/D))
    k = (quantizer[i]*D) + (D/2)
    z = int(10*k)
    quantizergray.append(gray(z))
    print(x)
    print(k)
    print(quantizer[i])
    print('/////////////////////////')
    receiver.append((quantizer[i]*D) + (D/2))
    q.append(receiver[i] - x)
    if i < 10 :
        mean1 = mean1 + yvalues[i]
        meanq1 = meanq1 + q[i]
    if i < 20 :
        mean2 = mean2 + yvalues[i]
        meanq = meanq2 + q[i] 
    i = i + 1
    j = j + Ts1
    if j > 0.00041 :
        break
print(i)
print('                    ')
print(quantizer)
print('                    ')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('                    ')
print(yvalues)
print('                    ')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('                    ')
print(receiver)
print('                    ')
plt.stem(t1, quantizer)
plt.title('QUANTIZER - Figure of y(t) with sampling frequency fs1 = 20fm and quantized in scale of x20')
plt.xlabel('time(s)')
plt.ylabel('y1(V)')
plt.grid(True)
plt.show()
plt.stem(t1, receiver)
plt.title('RECEIVER - Figure of y(t) with sampling frequency fs1 = 20fm and quantized in normal scale')
plt.xlabel('time(s)')
plt.ylabel('y1(V)')
plt.grid(True)
plt.show()
plt.stem(t1, quantizergray)
plt.title('QUANTIZERGRAY - Figure of y(t) with sampling frequency fs1 = 20fm and quantized and GRAY CODE in scale of x10')
plt.xlabel('time(s)')
plt.ylabel('y1(V)')
plt.grid(True)
plt.show() 
mean1 = mean1 / 10
mean2 = mean2 / 20
meanq1 = meanq1 / 10
meanq2 = meanq2 / 20
variance1 = 0
variance2 = 0
varianceq1 = 0
varianceq2 = 0
for k in range (0,10) :
    variance1 = variance1 + (yvalues[k] - mean1)**2
    varianceq1 = varianceq1 + (q[k] - meanq1)**2
print(varianceq1/9)
for k in range (0,20) :
    variance2 = variance2 + (yvalues[k] - mean2)**2
    varianceq2 = varianceq2 + (q[k] - meanq2)**2
print(varianceq1/19)
snr10 = variance1 / varianceq1
print(snr10)
snr20 = variance2 / varianceq2
print(snr20)
for i in range (0,41) :
    x = int(quantizergray[i])
    quantizergray[i] = x
binarysignal = []
print('                    ')
print(quantizergray)
print('                    ')
for i in range (0,41) : 
    w = quantizergray[i]
    if w == 1101 :
        binarysignal.append(1)
        binarysignal.append(0)
        binarysignal.append(1)
        binarysignal.append(1)
        binarysignal.append(0)
    elif w == 110 :
        binarysignal.append(0)
        binarysignal.append(1)
        binarysignal.append(1)
        binarysignal.append(0)
        binarysignal.append(0)
    elif w == 111 :
        binarysignal.append(1)
        binarysignal.append(1)
        binarysignal.append(1)
        binarysignal.append(0)
        binarysignal.append(0)
    elif w == 0 :
        binarysignal.append(0)
        binarysignal.append(0)
        binarysignal.append(0)
        binarysignal.append(0)
        binarysignal.append(0)
    else :
        binarysignal.append(1)
        binarysignal.append(0)
        binarysignal.append(0)
        binarysignal.append(0)
        binarysignal.append(0)
for i in range (0,205) :
    if (i%5) == 0 :
        print('@@@')
    print(binarysignal[i])
A = 5
period = 1
for i in range (0,50) :
    m = i*period
    n = (i+1)*period
    t = np.arange(m, n,0.01)
    if binarysignal[i] == 1 :
        plt.plot(t, A*signal.square(t, duty=1), 'b')
    else :
        plt.plot(t, -A*signal.square(t, duty=1), 'b')
plt.title('Bit flow for one period of thw signal')
plt.xlabel('time(mseconds)')
plt.ylabel('Volt(V)')
plt.grid(True)
plt.show()


# In[ ]:




