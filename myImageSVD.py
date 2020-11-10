#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


chis = [5, 10, 50, 100, 500]


# In[3]:


img1 = Image.open('./20200408030719.jpg')
img2 = Image.open('./20200511143422.jpg')
img1_gray = img1.convert('L')
img2_gray = img2.convert('L')


# In[4]:


#img1_gray.show(title = 'Original')
#img2_gray.show(title = 'Original')


# In[5]:


img1.save('./img1.jpg')
img2.save('./img2.jpg')
img1_gray.save('./img1_gray.jpg')
img2_gray.save('./img2_gray.jpg')


# In[6]:
f = open('result.txt','w')

array1 = np.array(img1_gray, dtype = float)
array2 = np.array(img2_gray, dtype = float)
f.write('Array1 shape:' + repr(array1.shape) + '\n')
f.write('Array2 shape:' + repr(array2.shape) + '\n')


# In[8]:


u1, s1, vt1 = np.linalg.svd(array1, full_matrices = False)
u2, s2, vt2 = np.linalg.svd(array2, full_matrices = False)

for chi in chis:
    u1_tr = u1[:, :chi]
    s1_tr = s1[:chi]
    vt1_tr = vt1[:chi, :]
    u2_tr = u2[:, :chi]
    s2_tr = s2[:chi]
    vt2_tr = vt2[:chi, :]
    array1_tr = np.dot(np.dot(u1_tr, np.diag(s1_tr)), vt1_tr)
    array2_tr = np.dot(np.dot(u2_tr, np.diag(s2_tr)), vt2_tr)
    normalized_dist1 = np.sqrt(np.sum((array1 - array1_tr)**2)) / np.sqrt(np.sum(array1**2))
    normalized_dist2 = np.sqrt(np.sum((array2 - array2_tr)**2)) / np.sqrt(np.sum(array2**2))
    f.write("Low rank approximation with chi = "+repr(chi) + '\n')
    f.write("Normalized distance 1:"+repr(normalized_dist1) + '\n')
    f.write("Normalized distance 2:"+repr(normalized_dist2) + '\n')

# In[9]:


    img1_gray_tr = Image.fromarray(np.uint8(np.clip(array1_tr, 0, 255)))
    img2_gray_tr = Image.fromarray(np.uint8(np.clip(array2_tr, 0, 255)))
    img1_gray_tr.save('./img1_gray_truncated_chi'+repr(chi)+'.jpg')
    img2_gray_tr.save('./img2_gray_truncated_chi'+repr(chi)+'.jpg')


# In[15]:


    s1_normalized = s1 / np.sqrt(np.sum(s1**2))
    s2_normalized = s2 / np.sqrt(np.sum(s2**2))


# In[21]:

colors = ['indianred', 'darkred', 'firebrick', 'red', 'maroon']

output_sv1 = len(s1)

plt.figure()
plt.title('Singular Value Spectrum of the image 1')
plt.plot(np.arange(output_sv1), s1_normalized, 'o', label = 'normalized singular values')
for i, chi in enumerate(chis):
    plt.axvline([chi], 0, 1, c = colors[i], linestyle = 'dashed', label = r'$\chi=${}'.format(chi))
plt.xlabel('Index')
plt.ylabel(r'$\sigma$')
plt.yscale('log')
plt.xticks(np.arange(0, output_sv1//100*100 + 100, 100))
plt.legend()
#plt.show()
plt.savefig('SVSpectrumImg1.png')
plt.close()

# In[22]:


output_sv2 = len(s2)

plt.figure()
plt.title('Singular Value Spectrum of the image 2')
plt.plot(np.arange(output_sv2), s2_normalized, 'o', label = 'normalized singular values')
for i, chi in enumerate(chis):
    plt.axvline([chi], 0, 1, c = colors[i], linestyle = 'dashed', label = r'$\chi=${}'.format(chi))
plt.xlabel('Index')
plt.ylabel(r'$\sigma$')
plt.yscale('log')
plt.xticks(np.arange(0, output_sv2//100*100 + 100, 100))
plt.legend()
#plt.show()
plt.savefig('SVSpectrumImg2.png')
plt.close()

# In[ ]:


f.close()

