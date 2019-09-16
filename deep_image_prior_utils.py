import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import tensorflow as tf

def cifar10_grid(X,Y,n_row,n_col,cifar10_class_name=None,resize=None,figsize=(10,10),interpolation="nearest"): 
    """
    Args :
    X : (N,32,32,3)
    Y : (N,1)
    
    Returns :
    Plot multiple images into grid (n_row,n_col) with title index of image and class.
    """
    if cifar10_class_name is None:
        cifar10_class_name = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck',
        }
         
         
    fig = plt.figure(figsize=figsize)   

    for i in range(n_row*n_col):
        a = fig.add_subplot(n_row, n_col, i+1)
        a.set_title("index : %d , Class %d (%s)" % (i,Y[i][0], cifar10_class_name[Y[i][0]]),fontsize=7)
        a.xaxis.set_tick_params(labelsize=1)
        a.yaxis.set_tick_params(labelsize=1)
        if resize is not None:
            plt.imshow(cv2.resize(X[i],resize),interpolation=interpolation)
            
        else:
            plt.imshow(X[i],interpolation=interpolation)
        
        
    plt.show()

def get_one_hot_vector(Y,num_classes,dtype=np.int32):
    """
    Args :
    Y : (N,1)
    num_classes : An integer
    
    Returns:
    A vector of shape (N,num_classes), where each row is a vector of shape (num_classes,) and it is all 0 but 1 at Y[i]
    """
    
    return np.squeeze(np.eye(num_classes)[Y]).astype(dtype)

def load_image(path):
    image = Image.open(path)
    return image
    
def pil_to_np(image_PIL):
    """
    Args :
    From h*w*c [0...255] to h*w*c [0...1]
    """
    array = np.array(image_PIL)
    
    if len(array.shape) != 3:
        array = array[None,...]
        
    else:
        array = array.transpose(2,0,1)
        
    return array.astype(np.float32)/255.
    
    
def np_to_pil(image_np):
    """
    Args :
    From h*w*c [0...1] to h*w*c [0...255]
    """
    array = np.clip(image_np*255,0,255).astype(np.uint8)
    
    if image_np.shape[0] == 1:
        array = array[0]
    
    else:
        array = array.transpose(1, 2, 0)
    
    return Image.fromarray(array)
    
    
def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped
        
def get_image(path,output_size=-1):
    image = load_image(path)
    
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
        
    if output_size[0] != -1 and image.size != output_size:
        if output_size[0] > image.size[0]:           
            image = image.resize(output_size, Image.BICUBIC)
        else:
            image = image.resize(output_size, Image.ANTIALIAS)

    image_np = pil_to_np(image)
    
    return image,image_np
    
    
def fill_noise(x,noise_type):
    
    if noise_type == 'u':
        return np.random.uniform(size=x.shape)
        
    elif noise_type == 'n':
        return np.random.normal(size=x.shape)
    
def get_noise(input_depth,method,spatial_size,noise_type='u',var=0.1):
    """
    
    Returns :
    numpy array of shape (1,spatial_size[0],spatial_size[1],input_depth)
    """
    assert method in ['noise','meshgrid']    
      
    if isinstance(spatial_size,int):
        spatial_size = (spatial_size,spatial_size)
        
    if method == 'noise':
        shape = [1,input_depth,spatial_size[0],spatial_size[1]]
        net_input = np.zeros(shape)
        net_input = fill_noise(net_input,noise_type)
        net_input *= var   # (1,input_depth,spatial_size[0],spatial_size[1])
        
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,...], Y[None,...]])
        net_input = meshgrid[None,...]  # (1,input_depth,spatial_size[0],spatial_size[1])
        
        
    return net_input  # (1,input_depth,spatial_size[0],spatial_size[1])
    
    
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """   
       
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
    
    
def load_LR_HR_imgs_sr(fname, imsize, factor, enforse_div32=None):
    '''Loads an image, resizes it, center crops and downscales.
    Args: 
        fname: path to the image
        imsize: new size for the image, -1 for no resizing
        factor: downscaling factor
        enforse_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
    '''
    img_orig_pil, img_orig_np = get_image(fname, -1)

    if imsize != -1:
        img_orig_pil, img_orig_np = get_image(fname, imsize)
        
    # For comparison with GT
    if enforse_div32 == 'CROP':
        new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, 
                    img_orig_pil.size[1] - img_orig_pil.size[1] % 32)

        bbox = [
                (img_orig_pil.size[0] - new_size[0])/2, 
                (img_orig_pil.size[1] - new_size[1])/2,
                (img_orig_pil.size[0] + new_size[0])/2,
                (img_orig_pil.size[1] + new_size[1])/2,
        ]

        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)
    else:
        img_HR_pil, img_HR_np = img_orig_pil, img_orig_np
        
    LR_size = [
               img_HR_pil.size[0] // factor, 
               img_HR_pil.size[1] // factor
    ]

    img_LR_pil = img_HR_pil.resize(LR_size, Image.ANTIALIAS)
    img_LR_np = pil_to_np(img_LR_pil)

    print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str (img_LR_pil.size)))

    return {
                'orig_pil': img_orig_pil,
                'orig_np':  img_orig_np,
                'LR_pil':  img_LR_pil, 
                'LR_np': img_LR_np,
                'HR_pil':  img_HR_pil, 
                'HR_np': img_HR_np
           }
           
           
def get_bernoulli_mask(for_image, zero_fraction=0.95):
    img_mask_np=(np.random.random_sample(size=pil_to_np(for_image).shape) > zero_fraction).astype(int)
    img_mask = np_to_pil(img_mask_np)
    
    return img_mask
    
    
def get_inpaining_network_input(image_path,mask_path,input_depth,method,imsize=-1,dim_div_by=64,noise_type='u',var=0.1,noise_distribution='uniform',reg_noise_std=0,dtype=np.float32):

    img_pil, img_np = get_image(image_path, imsize)                # e.g (512,512),(3,512,512)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)       # (512,512),(1,512,512)

    img_mask_pil = crop_image(img_mask_pil, dim_div_by)   # (512,512)
    img_pil      = crop_image(img_pil,      dim_div_by)   # (512,512)

    img_np      = pil_to_np(img_pil)                      # (3,512,512)
    img_mask_np = pil_to_np(img_mask_pil)                 # (1,512,512)
    
    img_array = img_np[None,...]             # (1,3,512,512)
    img_mask = img_mask_np[None,...]         # (1,1,512,512) 
    spatial_size = img_np.shape[1:]          # (512,512)
    noise = get_noise(input_depth,method,spatial_size,noise_type=noise_type,var=var).astype(dtype)  #(1,input_depth,spatial_size[0],spatial_size[1])
    shape = noise.shape
       
    noise_input = noise    #(1,input_depth,512,512)
    if reg_noise_std > 0:
        
        if noise_distribution == 'uniform':
            init = np.random.uniform(size=shape)
        
        elif noise_distribution == 'normal':
            init = np.random.normal(size=shape) 
    
        noise_input = noise + reg_noise_std*init    #(1,input_depth,512,512)
    
    noise_input = noise_input.astype(dtype)
    perm = (0,2,3,1)
    return img_array.transpose(perm),img_mask.transpose(perm),noise_input.transpose(perm)  #(1,512,512,3),(1,512,512,1),(1,512,512,input_depth)
    
   
### torch replicationpad2d
def replication_pad2d(inputs,pad_len,data_format='NHWC'):
    """
    Args :
    inputs : Assumes that it has a shape (N,C,H,W) or (N,H,W,C)
    pad_len : The length of paddings.
    
    Returns :
    Padded of inputs, which has a shape (N,C,H+2*pad_len,W+2*pad_len) or (N,H+2*pad_len,W+2*pad_len,C)
    """
    assert data_format in ['NHWC','NCHW']
    
    if data_format == 'NHWC':
        inputs = tf.transpose(inputs,(0,3,1,2))
    
    i = 0    
    paddings_up = tf.constant([[0, 0],[0,0],[1,0],[0,0]])
    paddings_down = tf.constant([[0, 0],[0,0],[0,1],[0,0]])
    paddings_left = tf.constant([[0, 0],[0,0],[0,0],[1,0]])
    paddings_right = tf.constant([[0, 0],[0, 0],[0,0],[0,1]])    
    
    def pad_one_side(inputs,paddings,i,pad_len):
        while i < pad_len:
            inputs = tf.pad(inputs, paddings,"SYMMETRIC")
            i += 1
        
        return inputs
        
    pad_up = pad_one_side(inputs,paddings_up,i,pad_len)
    pad_down = pad_one_side(pad_up,paddings_down,i,pad_len)
    pad_left = pad_one_side(pad_down,paddings_left,i,pad_len)
    pad_right = pad_one_side(pad_left,paddings_right,i,pad_len)  # (N,C,H+2*pad_len,W+2*pad_len)
    
    if data_format == 'NHWC':
        pad_right = tf.transpose(pad_right,(0,3,1,2))    # (N,H+2*pad_len,W+2*pad_len,C)
    
    return pad_right
        
        
#### Activation functions 

def swish(x):
    return x*tf.nn.sigmoid(x)


def activation_fns(act_fn_type='leaky_relu'):
    
    if act_fn_type == 'leaky_relu':
        # (features,alpha=0.2,name=None)
        return tf.nn.leaky_relu
        
    elif act_fn_type == 'swish':
        return swish
        
    elif act_fn_type == 'elu':
        # (features,name=None)
        return tf.nn.elu
        
    elif act_fn_type == 'relu':
        return tf.nn.relu

################

def get_total_num_parameters(train_vars=None):
    
    if train_vars is None:
        train_vars = tf.trainable_variables()
    return np.sum([np.prod(var.get_shape().as_list()) for var in train_vars])

########### Losses

def total_variation_loss(x,beta=0.5):
    """
    Args :
    x : 4D tensor (batch,h,w,c) or 3D tensor (h,w,c)
    
    Returns :
    If x is 4D, then return a 1D float tensor of shape (batch)
    If x is 3D, then return a scalar float.
    """
    loss = tf.reduce_sum(tf.image.total_variation(x))
    return loss
    