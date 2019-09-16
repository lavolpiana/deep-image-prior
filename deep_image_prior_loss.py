import os
import numpy as np
import tensorflow as tf

all_losses = {'mse': tf.losses.mean_squared_error,
'smoothL1': tf.losses.huber_loss,
'L1': tf.losses.absolute_difference}

def gram_matrix(x):
    """
    Args :
    x : 4D tensor (batch,h,w,c)
    """
    b,h,w,c = tf.shape(x)
    features = tf.reshape(x,[b,c,w*h])
    features_t = tf.transpose(features,[1,2])   #(b,w*h,c)
    gram = tf.matmul(features,features_t)/(h*w*c)  #(b,c,c)
    return gram
    
def features(x):
    return x
    
all_features = {'gram_matrix':gram_matrix,'features':features}




