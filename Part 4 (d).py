#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
import binascii
get_ipython().run_line_magic('matplotlib', 'inline')
def gray(n) : 
    z = n ^ n >> 1
    w = '{0:08b}'.format(z) 
    return w
def gray_to_binary(n):
    #Convert Gray codeword to binary and return it."""
    n = int(n, 2) # convert to int
 
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
 
    # bin(n) returns n's binary representation with a '0b' prefixed
    # the slice operation is to remove the prefix
    return bin(n)[2:]
f = open("C:\\shannon_odd.txt", "rt")
signal = []
signal = f.read()
print("The original signal")
print(type(signal))
print(signal)
print(" ")
print(" ")
signalint = []
for i in range(0, len(signal)) :
    signalint.append(ord(signal[i]))
print("The original signal in ASCII code")
print(type(signalint))
print(type(signalint[1]))
print(signalint)
print(" ")
print(" ")
signalbin = []
for i in range(0, len(signalint)) :
    signalbin.append('{0:08b}'.format(signalint[i]))
print("The original signal in ASCII Binary code")
print(type(signalbin))
print(type(signalbin[1]))
print(signalbin)
print(" ")
print(" ")
signalgray = []
for i in range(0, len(signalint)) :
    d = gray(signalint[i])
    signalgray.append(d)
print("The original signal in ASCII Binary Gray code")
print(type(signalgray))
print(type(signalgray[1]))
print(signalgray)
print(" ")
print(" ")
x = np.arange(0, len(signalgray))
plt.stem(x,signalgray)
plt.title("The quantized original signal in ASCII Binary Gray code")
plt.xlabel("Poistion of its character")
plt.ylabel("The quantized binary gray respresentation of its character")
plt.show()
bitflow = []
for i in range (0,len(signal)) :
    for j in range (0,8):
        bitflow.append(int(signalgray[i][j]))
print(" ")
print(" ")
print("The bitflow : ")
print(bitflow)


A = 1
Tb = 0.25
Eb = (A**2)*Tb
sqEb = np.sqrt(Eb)
Constellation = []
base = [np.complex(sqEb,sqEb), np.complex(sqEb,-sqEb), np.complex(-sqEb,sqEb), np.complex(-sqEb,-sqEb)]
x = [x.real for x in base]
y = [y.imag for y in base]
plt.plot(x,y,'or')
plt.title('The basic symbols of QPSK constellation diagram')
plt.show()
for i in range (0,len(bitflow)) :
    d = int(bitflow[i])
    bitflow[i] = d
    if bitflow[i] == 0 :
        bitflow[i] = -1
print("The bitflow : ")
print(bitflow)
for i in range (0, len(bitflow),2) :
    if (bitflow[i] == -1) and (bitflow[i+1] == -1) :
        Constellation.append(np.complex(-sqEb,-sqEb))
    if (bitflow[i] == -1) and (bitflow[i+1] == 1) :
        Constellation.append(np.complex(-sqEb,sqEb))
    if (bitflow[i] == 1) and (bitflow[i+1] == -1) :
        Constellation.append(np.complex(sqEb,-sqEb))
    if (bitflow[i] == 1) and (bitflow[i+1] == 1) :
        Constellation.append(np.complex(sqEb,sqEb))
k = [k.real for k in Constellation]
m = [m.imag for m in Constellation]
plt.plot(k,m,'*b')
plt.title('The Constellation diagram for the bit flow of the text')
plt.show()


No1 = Eb / (10**(5/10))
No2 = Eb / (10**(15/10))
X1 = np.random.normal(0, np.sqrt(No1), len(bitflow))
X2 = np.random.normal(0, np.sqrt(No2), len(bitflow))
constellation1 = []
constellation2 = []
estimate1 = []
estimate2 = []
for i in range (0,len(bitflow),2) :
    Esymbol11 = np.sqrt(((A*bitflow[i] + X1[i])**2)*Tb)
    Esymbol12 = np.sqrt(((A*bitflow[i+1] + X1[i+1])**2)*Tb)
    if ((A*bitflow[i] + X1[i]) > 0) and ((A*bitflow[i+1] + X1[i+1]) > 0):
        constellation1.append(np.complex(Esymbol11, Esymbol12))
        estimate1.append(1)
        estimate1.append(1)
    if ((A*bitflow[i] + X1[i]) > 0) and ((A*bitflow[i+1] + X1[i+1]) < 0):
        constellation1.append(np.complex(Esymbol11, -Esymbol12))
        estimate1.append(1)
        estimate1.append(-1)
    if ((A*bitflow[i] + X1[i]) < 0) and ((A*bitflow[i+1] + X1[i+1]) > 0):
        constellation1.append(np.complex(-Esymbol11, Esymbol12))
        estimate1.append(-1)
        estimate1.append(1)
    if ((A*bitflow[i] + X1[i]) < 0) and ((A*bitflow[i+1] + X1[i+1]) < 0):
        constellation1.append(np.complex(-Esymbol11, -Esymbol12))
        estimate1.append(-1)
        estimate1.append(-1)
    Esymbol21 = np.sqrt(((A*bitflow[i] + X2[i])**2)*Tb)
    Esymbol22 = np.sqrt(((A*bitflow[i+1] + X2[i+1])**2)*Tb)
    if ((A*bitflow[i] + X2[i]) > 0) and ((A*bitflow[i+1] + X2[i+1]) > 0):
        constellation2.append(np.complex(Esymbol21, Esymbol22))
        estimate2.append(1)
        estimate2.append(1)
    if ((A*bitflow[i] + X2[i]) > 0) and ((A*bitflow[i+1] + X2[i+1]) < 0):
        constellation2.append(np.complex(Esymbol21, -Esymbol22))
        estimate2.append(1)
        estimate2.append(-1)
    if ((A*bitflow[i] + X2[i]) < 0) and ((A*bitflow[i+1] + X2[i+1]) > 0):
        constellation2.append(np.complex(-Esymbol21, Esymbol22))
        estimate2.append(-1)
        estimate2.append(1)
    if ((A*bitflow[i] + X2[i]) < 0) and ((A*bitflow[i+1] + X2[i+1]) < 0):
        constellation2.append(np.complex(-Esymbol21, -Esymbol22))
        estimate2.append(-1)
        estimate2.append(-1)

w = [w.real for w in constellation1]
k = [k.imag for k in constellation1]
plt.plot(w, k , "*b")
x = [x.real for x in base] 
y = [y.imag for y in base] 
plt.plot(x, y, "or")
plt.title('Constellation diagram of bit flow QPSK, (π/4) Gray coding, 2-D AWGN and Eb / No equal 5dB')
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
plt.title('Constellation diagram of bit flow QPSK, (π/4) Gray coding, 2-D AWGN and Eb / No equal 15dB')
plt.xlabel('In-phase component')
plt.ylabel('Quadrature component')
plt.grid(True)
plt.show()

print('%%%%%%%%%%%%%%%')
print(estimate1)
print('%%%%%%%%%%%%%%%')
print(' ')
print(' ')
print(' ')
print(estimate2)
print('%%%%%%%%%%%%%%%')
errors1 = 0
errors2 = 0
for i in range (0,len(bitflow),2) :
    if (bitflow[i] != estimate1[i]) and (bitflow[i+1] != estimate1[i+1]):
        errors1 = errors1 + 2
    if (bitflow[i] == estimate1[i]) and (bitflow[i+1] != estimate1[i+1]):
        errors1 = errors1 + 1
    if (bitflow[i] != estimate1[i]) and (bitflow[i+1] == estimate1[i+1]):
        errors1 = errors1 + 1
    if (bitflow[i] != estimate2[i]) and (bitflow[i+1] != estimate2[i+1]):
        errors2 = errors2 + 2
    if (bitflow[i] == estimate2[i]) and (bitflow[i+1] != estimate2[i+1]):
        errors2 = errors2 + 1
    if (bitflow[i] != estimate2[i]) and (bitflow[i+1] == estimate2[i+1]):
        errors2 = errors2 + 1
BER1 = errors1 / len(bitflow)
BER2 = errors2 / len(bitflow)
print('The Bit Error Ration for bitflow QPSK, (π/4) Gray coding, 2-D AWGN and Eb / No equal 5dB is ', BER1, ' .')
print('The Bit Error Ration for bitflow QPSK, (π/4) Gray coding, 2-D AWGN and Eb / No equal 15dB is ', BER2, ' .')

recreate1 = []
recreate2 = []
for i  in range(0,len(estimate1)):
    if (estimate1[i] == (-1)):
        estimate1[i] = 0
for i  in range(0,len(estimate1)):
    if (estimate2[i] == (-1)):
        estimate2[i] = 0

for i in range (0,len(estimate1),8) :
    d = ""
    j = i
    while j < i + 8 :
        d = d + str(estimate1[j])
        j = j + 1
    recreate1.append(gray_to_binary(d))
for i in range (0,len(estimate2),8) :
    d = ""
    j = i
    while j < i + 8 :
        d = d + str(estimate2[j])
        j = j + 1
    recreate2.append(gray_to_binary(d))
print('#############')
print(type(recreate1))
print(type(recreate1[1]))
print(recreate1)
print('#############')
print(type(recreate2))
print(type(recreate2[1]))
print(recreate2)
print('#############')

string1 = ""
string2 = ""
for i in range(0,len(recreate1)):
    string1 = string1 + chr(int(recreate1[i],2))
for i in range(0,len(recreate2)):
    string2 = string2 + chr(int(recreate2[i],2))
print(string1)
print(string2)
file1 = open("shannon_odd_one.txt", "a")
file1.write(string1)
file1.close()
file2 = open("shannon_odd_two.txt", "a")
file2.write(string2)
file2.close()
#print(chr(int('01010111',2))) SOS !!


# In[ ]:




