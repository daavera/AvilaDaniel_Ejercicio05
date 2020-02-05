
# coding: utf-8

# In[127]:


import numpy as np
import matplotlib.pyplot as plt


# In[128]:


N = 1000
obs = ["s","c", "c", "c", "c"]
num_s = len(np.where(obs == "s"))
num_c = len(obs) - num_s


# In[129]:


H = np.linspace(1E-5,1-1E-5, N)
p_H = np.ones(N)
p_obs_H = (H**num_c)*((1-H)**num_s)
p_obs = np.trapz(p_obs_H,H)


# In[130]:


p_H_obs = p_obs_H*p_H/p_obs


# In[131]:


L = np.log(p_H_obs)

dH = H[1] - H[0]
d1_L = (L[:-1]-L[1:])/dH
d1_zeros = np.where(np.abs(d1_L)==np.min(np.abs(d1_L)))
H_max = H[d1_zeros[0][0]]


# In[132]:


d2_L = (d1_L[:-1]-d1_L[1:])/dH
sigma = 1/np.sqrt(-d2_L[d1_zeros][0])


# In[133]:


def gaussian(x, mu, sig):
    return (1/np.sqrt(2*np.pi*(sig**2)))*np.exp(-0.5*((x-mu)/sigma)**2)
gaussian = gaussian(H,H_max,sigma)


# In[136]:


plt.plot(H, p_obs_H*p_H/p_obs)
plt.plot(H, gaussian, "--")
plt.title("H = " + str('{:2f}'.format(float(H_max))) + "$\pm$ " + str('{:2f}'.format(float(sigma))))
plt.xlabel("H")
plt.ylabel("P(H | {obs})")
plt.savefig("coins.png")

