#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
fm = 5000
fs = 5 * fm
fs1 = 20 * fm 
fs2 = 100 * fm
Ts = 1/ fs
Ts1 = 1 / (fs1)
Ts2 = 1 / (fs2)
AM = 5
t = np.arange(0,0.00041,0.0000001)   # start,stop,step
t1 = np.arange(0,0.00041,Ts1)   # start,stop,step
t2 = np.arange(0,0.00041,Ts2)   # start,stop,step
ts = np.arange(0,0.00041,Ts)   # start,stop,step
y = np.cos(2*np.pi*fm*t)*np.cos(2*np.pi*(AM + 2)*fm*t)
y1 = np.cos(2*np.pi*fm*t1)*np.cos(2*np.pi*(AM + 2)*fm*t1)
y2 = np.cos(2*np.pi*fm*t2)*np.cos(2*np.pi*(AM + 2)*fm*t2)
ys = np.cos(2*np.pi*fm*ts)*np.cos(2*np.pi*(AM + 2)*fm*ts)
plt.plot(t, y)
plt.title('Figure of y(t)')
plt.xlabel('time(s)')
plt.ylabel('y(V)')
plt.show()
plt.stem(t1, y1)
plt.title('Figure of y(t) with sampling frequency fs1 = 20fm')
plt.xlabel('time(s)')
plt.ylabel('y1(V)')
plt.show()
plt.stem(t2, y2)
plt.title('Figure of y(t) with sampling frequency fs2 = 100fm')
plt.xlabel('time(s)')
plt.ylabel('y2(V)')
plt.show()
plt.stem(t1, y1, 'r', markerfmt ='ro')
plt.stem(t2, y2, 'g', markerfmt ='gd')
plt.title('Compare the two sampled signals')
plt.xlabel('time(s)')
plt.ylabel('y1(V),y1(V)')
plt.show()
plt.stem(ts, ys)
plt.title('Figure of y(t) with sampling frequency fs = 5fm')
plt.xlabel('time(s)')
plt.ylabel('ys(V)')
plt.show()


# In[ ]:




