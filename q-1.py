#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import pandas as pd
eps = np.finfo(float).eps
import sys
from matplotlib.pyplot import imshow


# In[89]:


def sigmoid(data):
    return 1 / (1 + np.exp(np.negative(data)))

def calc_softmax(a,b=0, t=1.0):
    e = np.exp(a / t)
    ans = e / np.sum(e)
    return ans

def calc_convolve(data, fltr):  
    res = np.multiply(data, fltr)
    return res

def calc_matmul(x,y):
    return np.matmul(x,y)

def ReLU(a):
    return a * (a > 0)


# In[90]:


IMAGE_PATH = '/home/aishwarya/CSIS/SMAI/SMAI_assig/a-6/test_img.png'

FC_Convolution_Filter_Count = 120
POOLING_DIM = 2

def create_filters(layer_count):
    filter_depth = 0
    if layer_count == 1:
        filter_depth = 6
    elif layer_count == 2:
        filter_depth = 16
    elif layer_count == 3:
        filter_depth = FC_Convolution_Filter_Count
    else:
        print("Invalid layer count")

    filt_mtrx = np.random.randn(5,5, filter_depth)
    return filt_mtrx

def convolution(data, filter):
    data_row = data.shape[0]
    f_row, f_col, num_filters = filter.shape
    
    conv_result_dim = data_row - f_row + 1
    data_col = data.shape[1]
    data_channels = data.shape[2]
    
    i=0
    convolve_res = np.zeros((conv_result_dim, conv_result_dim, num_filters))

    # Breaking the data into sub-matrices of filter size and calling calc_convolve
    while i<num_filters:     
        for x_index in range(conv_result_dim):
            for y_index in range(conv_result_dim):
                ans=calc_convolve(data[x_index: x_index + f_row, y_index: y_index + f_col, i % data_channels],filter[:, :, i])
                convolve_res[x_index][y_index][i] = ans.sum()
        i+=1
#         im = Image.fromarray(convolve_result)
#         nm='test'+filter_index+'.png'
#         im.save(nm)

    return convolve_res

def imgshow(x,name):
    #x = x.transpose(2,0,1) 
    img = Image.fromarray(x,'RGB')
    img = img.resize((300,300))
    img.save(name)

def calc_weights_matrix(x,y):
#     or return np.random.randn(x,y)
    return np.random.uniform(low=-1, high=1, size=(x,y) )

def calc_max_pooling(data, sze=2):#, pool_size):
    data_row, data_col, result_layer = data.shape
    result_row= data_row // sze
    result_col= data_col // sze

    max_pool_result = np.zeros((result_row, result_col, result_layer))

    for l in range(result_layer):
        for r in range(0, data_row, 2):
            for c in range(0, data_col, 2):
                max_pool_result[r // sze][c // sze][l] = (data[r:r + sze, c:c + sze, l]).max()

    return max_pool_result


# In[91]:


image = Image.open(IMAGE_PATH)
image = image.resize((32,32), Image.ANTIALIAS)
image.save('resized_img_32x32.png')

get_ipython().magic(u'matplotlib inline')
imshow(np.asarray(image))


# In[103]:


# if __name__ == '__main__':
def CNN(act_fun="sigmoid",pooling_type="max_pooling"):
    # Lenet Architectue
    # INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC(Conv) => RELU => FC

    
    image_array = np.array(image)
    print "Image new Dimensions: ", image_array.shape

    # -------------------------- First Convolution Block -------------------------------------------
    # Creating first layer filter
    # apply convolution on original image with 6 diffrent filters
    print '\n\nFirst convolution:'
    filter_matrix = create_filters(1)
    print "Filter Dimensions: ", filter_matrix.shape
    
    imgshow(filter_matrix,'1_filter_matrix.png')
    plt.imshow(filter_matrix[0])
    
    # Convolution at first layer
    convolve_res = convolution(image_array, filter_matrix)
    print "Convolution Result Dimensions:", convolve_res.shape
    imgshow(convolve_res,'1_conv_result.png')
#     plt.imshow(convolve_res[0])
    
    if act_fun=="ReLU":
        # Applying ReLU activation function
        relu_result = ReLU(convolve_res)
        print "ReLU Result Dimensions:", relu_result.shape
        imgshow(relu_result,'1_relu_result.png')
    elif act_fun=="sigmoid":
        relu_result = sigmoid(convolve_res)
        print "sigmoid Result Dimensions:", relu_result.shape
        imgshow(relu_result,'1_sigmoid_result.png')


    if pooling_type=="max_pooling":
        # MaxPooling at the first convolution block
        pool_result = calc_max_pooling(relu_result, POOLING_DIM)
        imgshow(pool_result,'1_maxpool_result.png')
    print pool_result.shape , 'New image shape after 1st pooling'
    plt.imshow(pool_result[0])

    # --------------Second Convolution Block-----------------------------------------
    print '\n\nSecond convolution:'
    filter_matrix = create_filters(2)
    print "Filter Dimensions (2nd Block): ", filter_matrix.shape

    imgshow(filter_matrix,'2_filter_matrix.png')

    convolve_result = convolution(pool_result, filter_matrix)
    print "Convolution Result Dimensions (2nd Block):", convolve_result.shape

    imgshow(convolve_result,'2_conv_result.png')

    if act_fun=="ReLU":
        # Applying ReLU activation function
        relu_result = ReLU(convolve_result)
        print "ReLU Result Dimensions:", relu_result.shape
        imgshow(relu_result,'2_relu_result.png')
    elif act_fun=="sigmoid":
        relu_result = sigmoid(convolve_result)
        print "sigmoid Result Dimensions:", relu_result.shape
        imgshow(relu_result,'2_sigmoid_result.png')

        

    if pooling_type=="max_pooling":
        # MaxPooling at the first convolution block
        pool_result = calc_max_pooling(relu_result, POOLING_DIM)
        imgshow(pool_result,'2_maxpool_result.png')

    print pool_result.shape , 'New image shape after 2nd pooling'

    # -------------------- Convolution to give Fully Connected Layer-----------------------
    print '\n\nFully Connected Layer-----------------------[forward propagation]------------------------'
    
    input_matrix=np.array(pool_result.flatten().reshape(1, len(pool_result.flatten())))
    print 'current shape-',input_matrix.shape

    # -----------layer 1 -120-----------
    weight_input_hidden=calc_weights_matrix(400,120)
    
    hiddn_layer1=calc_matmul(input_matrix, weight_input_hidden)
    i=0
    while i<  len (hiddn_layer1[0]):
        hiddn_layer1[0][i]=sigmoid(hiddn_layer1[0][i])
        i+=1

    print '\noutput layer 1 shape-',hiddn_layer1.shape

    # -----------Layer 2- 84-----------
    input_mat=hiddn_layer1
    weight_input_hidden=calc_weights_matrix(120,84)
#     print 'weght matr',weight_input_hidden
    hiddn_layer2=calc_matmul(input_mat, weight_input_hidden)
    j=0
    while j<  len (hiddn_layer2[0]):
        hiddn_layer2[0][j]=sigmoid(hiddn_layer2[0][j])
        j+=1

    print '\noutput layer 2 shape-',hiddn_layer2.shape

    # -----------layer 3 -10-----------
    weight_input_hidden=calc_weights_matrix(84,10)
    output_layer=calc_matmul(hiddn_layer2, weight_input_hidden)
    k=0
    while k<  len (output_layer[0]):
        output_layer[0][k]=sigmoid(output_layer[0][k])
        k+=1
        
    print '\noutput layer 3 -',output_layer
    k=0
    while k<  len (output_layer[0]):
        output_layer[0][k]=calc_softmax(output_layer[0][k])
        k+=1

#     softmax_result = calc_softmax(output_nn[0, 0, :])
#     print(softmax_result)
    print '\noutput layer 3 shape-',output_layer.shape
    print '\nSOFTMAX:output layer 3 -',output_layer


# In[104]:


act_fun="ReLU"
CNN(act_fun,pooling_type="max_pooling")

