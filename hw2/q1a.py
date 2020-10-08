import numpy as np
import matplotlib.pyplot as plt
def SE(y,t):
    return 0.5*(y-t)**2
def HL(y,t,theta):
    hl = np.zeros(y.size)
    a = y-t
    for i in range(a.size):
        if np.abs(a[i])<=theta:
            hl[i] = 0.5*a[i]**2
        else:
            hl[i] = theta*(np.abs(a[i])-0.5*theta)
    return hl

y = np.linspace(0,2,100)
t = np.zeros(100)
se = SE(y,t)
hl1 = HL(y,t,1.0)
hl2 = HL(y,t,0.5)
hl3 = HL(y,t,0.1)
#print ("y is ",y," ","se is ", se)
plt.plot(y,se, label="SE")
plt.plot(y,hl1, label="HL with theta=1.0")
plt.plot(y,hl2, label="HL with theta=0.5")
plt.plot(y,hl3, label="HL with theta=0.1")
plt.legend()
plt.show()
