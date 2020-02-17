#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.misc import imresize


# In[ ]:





# In[2]:


def get_data(imgsamples, labels):
    data = {}

    for image in imgsamples:
        data[image] = []
        for label in labels:
            if image == label.split(',')[0].split('/')[-1]:
                data[image].append([int(label.split(',')[1]), int(label.split(',')[2]), 
                                    int(label.split(',')[3]), int(label.split(',')[4])])
    return data


# In[3]:


def showbox(img, boxes, h_r=1, w_r=1):
    t = img.copy()
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = x1//w_r, x2//w_r
        y1, y2 = y1//h_r, y2//h_r
        rect = Rectangle((x1,y1),x2-x1,y2-y1, fill=None, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    if len(t.shape) == 3:
        ax.imshow(t[:, :, 0], cmap = 'gray')
    else:
        ax.imshow(t, cmap = 'gray')


# In[6]:


def resize_img(img, output_h = 800, output_w = 600, output_c = 1):
    img_resized = imresize(img, (output_h, output_w)) 

    if output_c == 1 and len(img_resized.shape) == 3:
        img_resized = img_resized.mean(axis=2).astype(np.float32)
        img_resized = img_resized[:, :, np.newaxis]
    if output_c == 1 and len(img_resized.shape) == 2:
        img_resized = img_resized[:, :, np.newaxis]
        
    return img_resized


# In[7]:


def get_train_test(imgsamples, train_ratio = 0.90):

    idx = np.arange(len(imgsamples)) 
    np.random.shuffle(idx) 
    idx_1 = idx[:int(train_ratio*len(imgsamples))] 
    idx_2 = idx[int(train_ratio*len(imgsamples)):]
    img_train = [imgsamples[k] for k in idx_1]
    img_test = [imgsamples[k] for k in idx_2]
    
    return img_train, img_test


# In[8]:


def prepare_data(data, path, output_h = 800, output_w = 600, output_c = 1, need_resize = True):
    
    imgbatch = list(data.keys())
    labelbatch = [data[k] for k in imgbatch]
    
    X_data = []
    y_data = []
    
    for i in range(len(imgbatch)):
        img = plt.imread(path + imgbatch[i])
        
        if need_resize:
            img_resize = resize_img(img, output_h = output_h, output_w = output_w, output_c = output_c)
            h_r, w_r = img.shape[0]/output_h, img.shape[1]/output_w
            boxes = [[k[0]//w_r, k[1]//h_r, k[2]//w_r, k[3]//h_r] for k in labelbatch[i]]
            y1 = np.ones((output_h, output_w)) # object
            y2 = np.zeros((output_h, output_w)) # background
        else:
            img_resize = img
            boxes = [[k[0], k[1], k[2], k[3]] for k in labelbatch[i]]
            y1 = np.ones((img.shape[0], img.shape[1]))
            y2 = np.zeros((img.shape[0], img.shape[1]))
        
        
        for i in range(y1.shape[0]):
            for j in range(y1.shape[1]):
                for box in boxes:
                    if i >= box[1] and i<= box[3] and j >= box[0] and j <= box[2]:
                        y1[i, j] = 0
                        y2[i, j] = 1
        
        ##y1 = y1[:, :, np.newaxis]
        ##y2 = y2[:, :, np.newaxis]
        ##img_label = np.concatenate([y1, y2], axis = 2)
        img_label = np.stack((y1, y2), axis = 2)
            
        X_data.append(img_resize)
        y_data.append(img_label)
        
    X_data = np.stack(X_data, axis = 0)
    y_data = np.stack(y_data, axis = 0)
    return X_data, y_data


# In[ ]:





# In[ ]:





# In[ ]:




