#!/usr/bin/env python
# coding: utf-8

# # Imports




import matplotlib.pyplot as plt 
import numpy as np


# # Image chat.jpg 

# Reading image : 




chat = plt.imread('chat.jpg')


# Image plot :



plt.imshow(chat)
_ = plt.title('cat image before compression')



# The image has 3 channels (RGB) 605x605 pixels for each channel




chat.shape





plt.imshow(chat[:,:,0], cmap = 'gray')





plt.imshow(chat[:,:,1], cmap = 'gray')





plt.imshow(chat[:,:,2], cmap = 'gray')


# # SVD Compression 

# ## Channel 0 




U, Sigma, Vt = np.linalg.svd(chat[:,:,0])





Sigma.shape


# Note that Sigma is already sorted in descending order as shown in the documentation
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html




U.shape





Vt.shape





for k in range(10,len(Sigma),10): 
    compression_0 = U[:,:k] @ np.diag(Sigma[:k]) @ Vt[:k,:]
    plt.title(f'k ={k}')
    plt.imshow(compression_0, cmap = 'gray')
    plt.show()


# ## Channel 1




U, Sigma, Vt = np.linalg.svd(chat[:,:,1])





for k in range(10,len(Sigma),10): 
    compression_0 = U[:,:k] @ np.diag(Sigma[:k]) @ Vt[:k,:]
    plt.title(f'k ={k}')
    plt.imshow(compression_0, cmap = 'gray')
    plt.show()


# ##  Channel 2




U, Sigma, Vt = np.linalg.svd(chat[:,:,2])





for k in range(10,len(Sigma),10): 
    compression_0 = U[:,:k] @ np.diag(Sigma[:k]) @ Vt[:k,:]
    plt.title(f'k ={k}')
    plt.imshow(compression_0, cmap = 'gray')
    plt.show()


# ## Reconstruction of the image 




U0, Sigma0, Vt0 = np.linalg.svd(chat[:,:,0])
U1, Sigma1, Vt1 = np.linalg.svd(chat[:,:,1])
U2, Sigma2, Vt2 = np.linalg.svd(chat[:,:,2])

for k in range(10,len(Sigma0),10): 
    compression_0 = U0[:,:k] @ np.diag(Sigma0[:k]) @ Vt0[:k,:]
    compression_1 = U1[:,:k] @ np.diag(Sigma1[:k]) @ Vt1[:k,:]
    compression_2 = U2[:,:k] @ np.diag(Sigma2[:k]) @ Vt2[:k,:]
    compression = np.zeros((len(Sigma0), len(Sigma0), 3))
    compression[:,:,0] = compression_0
    compression[:,:,1] = compression_1
    compression[:,:,2] = compression_2
    compression = compression - np.min(compression)  # set min pixel value to 0
    compression = compression * 255/ np.max(compression) # set max pixel value to 255 
    compression = compression.astype('int')  # we want integer pixel values
    plt.title(f'k ={k}')
    plt.imshow(compression)
    plt.show()


# Notice that there are some color artifacts due the compression of each color channel independently

# # PSNR




k_values = []
psnr_values = []
for k in range(1,len(Sigma0),5): 
    compression_0 = U0[:,:k] @ np.diag(Sigma0[:k]) @ Vt0[:k,:]
    compression_1 = U1[:,:k] @ np.diag(Sigma1[:k]) @ Vt1[:k,:]
    compression_2 = U2[:,:k] @ np.diag(Sigma2[:k]) @ Vt2[:k,:]
    compression = np.zeros((len(Sigma0), len(Sigma0), 3))
    compression[:,:,0] = compression_0
    compression[:,:,1] = compression_1
    compression[:,:,2] = compression_2
    compression = compression - np.min(compression)  # set min pixel value to 0
    compression = compression * 255/ np.max(compression) # set max pixel value to 255 
    compression = compression.astype('int')  # we want integer pixel values
    
    Rmax2 = 255 **2
    e2 = np.linalg.norm(chat - compression)**2  /(605*605)
    psnr = 10 * np.log10(Rmax2 / e2)
    print(f'for k = {k} the PSNR = {psnr}')
    k_values.append(k)
    psnr_values.append(psnr)




plt.plot(k_values, psnr_values)
plt.xlabel('k')
plt.ylabel('PSNR')
plt.title('PSNR as a function of k')

