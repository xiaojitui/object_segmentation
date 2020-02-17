#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet import image_util

from prepare import *
#from __future__ import division, print_function
#%matplotlib inline


# In[19]:


################## Introduction ################## 

# X is in shape of [B, W, H, C]
# y is in shape of [B, W, H, classes]

# tf_unet.unet.Unet(channels=3, n_class=2, cost=u'cross_entropy', layers=3, features_root=16, cost_kwargs={}, **kwargs)
## parameters:
##        cost_kwargs – (optional) kwargs passed to the cost function. See Unet._get_cost for more options
##        cost_kwargs=dict(regularizer=0.001) - use L2 regularizer 
##        layers – number of layers in the net
##        features_root – number of features in the first layer, default = 16
##        filter_size – size of the convolution filter, default = 3
##        pool_size – size of the max pooling operation, default = 2


## modules:
## Unet.restore(sess, model_path)
## Unet.save(sess, model_path)


# tf_unet.unet.Trainer(net, batch_size=1, verification_batch_size=4, norm_grads=False, optimizer= u’momentum’, opt_kwargs={})
## parameters:
##      optimizer – (optional) name of the optimizer to use (momentum or adam)
##      opt_kwargs – (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

## modules:
## Trainder.train(data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=
##                False, write_graph=False, prediction_path=u’prediction’)
###               data_provider – callable returning training and verification data




# tf_unet.image_util.ImageDataProvider(search_path, a_min=None, a_max=None,
#                                      data_suffix=u’.tif’, mask_suffix=u’_mask.tif’,
#                                      shuffle_data=True, n_class=2)


# tf_unet.image_util.SimpleDataProvider(data, label, a_min=None, a_max=None,
#                                       channels=1, n_class=2)
## parameters:
##           data – data numpy array. Shape=[n, X, Y, channels]
##           label – label numpy array. Shape=[n, X, Y, classes]


# In[20]:


##################  example 1 ################## 
# net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)
# trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
# path = trainer.train(generator, "./unet_trained", training_iters=32, epochs=10, display_step=2)
# prediction = net.predict("./unet_trained/model.ckpt", x_test)
# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
# ax[0].imshow(x_test[0,...,0], aspect="auto")
# ax[1].imshow(y_test[0,...,1], aspect="auto")
# mask = prediction[0,...,1] > 0.9
# ax[2].imshow(mask, aspect="auto")


# In[21]:


##################  example 2 ################## 

# net = unet.Unet(channels=data_provider.channels, 
#                 n_class=data_provider.n_class, 
#                 layers=3, 
#                 features_root=64,
#                 cost_kwargs=dict(regularizer=0.001),
#                 )

# trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
# path = trainer.train(data_provider, "./unet_trained_bgs_example_data", 
#                      training_iters=32, 
#                      epochs=1, 
#                      dropout=0.5, 
#                      display_step=2)                

# prediction = net.predict(path, x_test)

# fig, ax = plt.subplots(1,3, figsize=(12,4))
# ax[0].imshow(x_test[0,...,0], aspect="auto")
# ax[1].imshow(y_test[0,...,1], aspect="auto")
# ax[2].imshow(prediction[0,...,1], aspect="auto")


# In[22]:


##################  general procedure ################## 
# 1. prepare X and y: X is in shape of [B, W, H, C], y is in shape of [B, W, H, classes]
# 2. net = unet.Unet()
# 3. trainer = unet.Trainer()
# 4. data_provider = image_util.SimpleDataProvider(X, y, channels=1, n_class=2)
# 5. path = trainer.train(data_provider,...)
# 6. prediction = net.predict(path, x_test) or "./unet_trained/model.ckpt"


# In[ ]:





# In[ ]:





# In[23]:


#### main


# In[24]:


def prepare_data():
    imgpath = './train_images/'
    imgfiles = os.listdir(imgpath)
    imgsamples = [k for k in imgfiles if k.endswith('.png')]

    with open('./annotate.txt') as f:
        labels = f.readlines()

    img_train, img_test = get_train_test(imgsamples, train_ratio = 0.95)
    data_train = get_data(img_train, labels)
    data_test = get_data(img_test, labels)
    
    X_train, y_train = prepare_data(data_train, imgpath, output_h = 572, output_w = 572)
    X_test, y_test = prepare_data(data_test, imgpath, output_h = 572, output_w = 572)
    
    for i in range(len(X_train)):
        X_train[i] = X_train[i] /255
        
    for i in range(len(X_test)):
        X_test[i] = X_test[i] /255 #(X_test[i] - 127.5)/127.5


# In[25]:


def vis_data(X_test, y_test, idx):
    #idx = 14
    fig, ax = plt.subplots(1, 3, figsize = (10, 10))
    ax[0].imshow(X_test[idx][:, :, 0], cmap = 'gray')
    ax[1].imshow(y_test[idx][:, :, 0], cmap = 'gray')
    ax[2].imshow(y_test[idx][:, :, 1], cmap = 'gray')


# In[26]:


def train():
    net = unet.Unet(channels=1, n_class=2, layers=4, features_root=64)
    trainer = unet.Trainer(net, batch_size=8, verification_batch_size=4, optimizer= 'adam')
    #data_provider = image_util.SimpleDataProvider(X_test, y_test)
    path = trainer.train(X_test, y_test, X_test, y_test, './pre_trained', training_iters=32, epochs=50, 
                         dropout=0.5, display_step=8, restore = True) #restore = True
    


# In[28]:


def predict(img):
    
    if img.shape == 2:
        img = img[np.newaxis, :, :, np.newaxis]
        
    if img.shape == 3:
        img = img[np.newaxis, :, :, :]
        
    net = unet.Unet(channels=1, n_class=2, layers=4, features_root=64)
    prediction = net.predict('./pre_trained/model.ckpt', img)
    
    return prediction


# In[29]:


def vis_prediction(X, y, idx):
    
    img_ = X[idx].copy()
    prediction = predict(img_)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(X[idx,...,0], cmap = 'gray')
    ax[1].imshow(y[idx,...,1],  cmap = 'gray')
    mask = prediction[0,...,1] > 0.5
    ax[2].imshow(mask,  cmap = 'gray')
    #ax[2].imshow(prediction[0,...,1], cmap = 'gray')


# In[ ]:





# In[ ]:


if __name__ == '__main__':
    train()


# In[ ]:





# In[ ]:





# In[ ]:




