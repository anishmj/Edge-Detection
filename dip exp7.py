#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('city.jpeg')
cv2.imshow("original",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


img = cv2.imread('city.jpeg')
scale = 10000 
width = int(img.shape[1] * scale / 100)  
height = int(img.shape[0] * scale / 100) 
dim = (width, height)  
# resize image  
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) 
plt.imshow(resized)
plt.show()


# In[3]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


img = cv2.GaussianBlur(gray,(3,3),0)
cv2.imshow("Gaussian blur",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("bgr2gray",gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
plt.figure(1)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
cv2.imshow("SobelX",sobelx)
cv2.waitKey(0)
cv2.destroAllWindows()


# In[7]:


sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.figure(1)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')


cv2.imshow("sobely",sobely)
cv2.waitKey(0)
cv2.destroyAllWindows


# In[8]:


sobel_xy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
cv2.imshow('sobel_xy',sobel_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


# LAPLACIAN EDGE DETECTOR AND TO SHOW THE DETECTED IMAGE

rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
laplacian_operator = cv2.Laplacian(rgb_image,cv2.CV_64F)
cv2.imshow('laplacian_operator',laplacian_operator)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


canny_edges = cv2.Canny(img,120,150)
plt.figure(1)
cv2.imshow("Canny Edges",canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows


# In[ ]:




