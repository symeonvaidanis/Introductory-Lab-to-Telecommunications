#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
def gray(n) : 
    z = n ^ n >> 1
    w = '{0:08b}'.format(z) 
    return w
# Helper function to flip the bit
def flip(c):
    return '1' if(c == '0') else '0';
# function to convert gray code
# string to binary string
def gray_to_binary(gray):
 
    binary = "";
 
    # MSB of binary code is same 
    # as gray code
    binary += gray[0];
 
    # Compute remaining bits
    for i in range(1, len(gray)):
         
        # If current bit is 0, 
        # concatenate previous bit
        if (gray[i] == '0'):
            binary += binary[i - 1];
 
        # Else, concatenate invert 
        # of previous bit
        else:
            binary += flip(binary[i - 1]);
 
    return binary;

samplerate, data = wavfile.read('C:\\soundfile1_lab2.wav')
print(samplerate)
print(data)
print(data.shape)
print(data.shape[0])
print(np.amax(data))
print(np.amin(data))
print(type(data))
print(type(data[34]))

times = np.arange(len(data))/float(samplerate)
plt.figure(figsize=(30, 20))
plt.fill_between(times, data) 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title("The audio signal")
plt.savefig('plot.png', dpi=100)
plt.show()
plt.plot(data)
plt.title("The audio signal")
plt.show()

R = 8
L = 2 ** R
D = (np.amax(data) - np.amin(data)) / L
print(D)
quantizer = np.empty(data.shape[0], dtype = float)
for i in range (0, data.shape[0]) :
    quantizer[i] = (np.floor(data[i]/D))*D + (D/2)
print(np.amax(quantizer), "  ", np.amin(quantizer))    
times1 = np.arange(len(data))/float(samplerate)
plt.figure(figsize=(30, 20))
plt.fill_between(times, quantizer) 
plt.xlim(times1[0], times1[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title("The quantized audio signal")
plt.savefig('plotquantized.png', dpi=100)
plt.show()
plt.plot(data)
plt.title("The quantized audio signal")
plt.show()

compare = np.unique(quantizer)
print(compare.shape[0])
print(np.amax(quantizer) / D)
print(np.amin(quantizer) / D)
for i in range(0, compare.shape[0]) :
    print(compare[i]/D)

print('%%%%%%%%%%%%')
quantizergray = []
for i in range (0, data.shape[0]) :
    d = int(quantizer[i] / D)
    quantizergray.append(gray(d))
print(type(quantizergray))
print(len(quantizergray))
print(len(quantizergray[3]))
plt.plot(quantizergray)
plt.title('quantizergray')
plt.show()
bitflow = []
for i in range (0,len(quantizergray)) :
    for j in range (0,8):
        bitflow.append(int(quantizergray[i][j]))
print('%%%%%%%%%%%%')
print(type(bitflow))
print(len(bitflow))
print(type(bitflow[5]))
plt.plot(bitflow)
plt.title('bitflow 0')
plt.show()
print('%%%%%%%%%%%%')

for i in range (0,len(bitflow)) :
    if bitflow[i] == 0 :
        bitflow[i] = -1
plt.plot(bitflow)
plt.title('bitflow -1')
plt.show()

print('%%%%%%%%%%%%')
A = 1
Tb = 0.25
Eb = (A**2)*Tb
sqEb = np.sqrt(Eb)
No1 = Eb / (10**(4/10))
No2 = Eb / (10**(14/10))
X1 = np.random.normal(0, np.sqrt(No1), len(bitflow))
X2 = np.random.normal(0, np.sqrt(No2), len(bitflow))
constellation1 = []
constellation2 = []
estimate1 = []
estimate2 = []
base = [np.complex(sqEb,sqEb), np.complex(sqEb,-sqEb), np.complex(-sqEb,sqEb), np.complex(-sqEb,-sqEb)]
x = [x.real for x in base]
y = [y.imag for y in base]
plt.plot(x,y,'or')
plt.title('The basic symbols of QPSK constellation diagram')
plt.show()
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
plt.title('Constellation diagram of bit flow QPSK, (π/4) Gray coding, AWGN and Eb / No equal 4dB')
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
plt.title('Constellation diagram of bit flow QPSK, (π/4) Gray coding, AWGN and Eb / No equal 14dB')
plt.xlabel('In-phase component')
plt.ylabel('Quadrature component')
plt.grid(True)
plt.show()

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
print('The Bit Error Ration for bitflow QPSK, (π/4) Gray coding, AWGN and Eb / No equal 4dB is ', BER1, ' .')
print('The Bit Error Ration for bitflow QPSK, (π/4) Gray coding, AWGN and Eb / No equal 14dB is ', BER2, ' .')

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
print(len(recreate1))
plt.plot(recreate1)
plt.title('recreate1')
plt.show()
print('#############')
print(type(recreate2))
print(type(recreate2[1]))
print(len(recreate2))
plt.plot(recreate2)
plt.title('recreate2')
plt.show()
print('#############')

audio1 = np.empty(len(recreate1), dtype = np.uint8)
audio2 = np.empty(len(recreate2), dtype = np.uint8)
for i in range(0,len(recreate1)) :
    audio1[i] = (int(recreate1[i],2) + 0.5)
    audio2[i] = (int(recreate2[i],2) + 0.5)
write("example1.wav", samplerate, audio1)
write("example2.wav", samplerate, audio2)
plt.plot(audio1)
plt.title('audio1')
plt.show()
plt.plot(audio2)
plt.title('audio2')
plt.show()


# In[ ]:




