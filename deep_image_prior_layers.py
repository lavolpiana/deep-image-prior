import numpy as np
import tensorflow as tf
from deep_image_prior_utils import *


######## Downsampler ##################

def get_kernel(factor,kernel_type,phase,kernel_width,support=None,sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width-1,kernel_width-1])
        
    else:
        kernel = np.zeros([kernel_width,kernel_width])
        
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1./(kernel_width * kernel_width)
        
    elif kernel_type == 'gauss': 
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        
        center = (kernel_width + 1.)/2.
        print(center, kernel_width)
        sigma_sq =  sigma * sigma
        
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center)/2.
                dj = (j - center)/2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))/(2. * np.pi * sigma_sq)                
                
    elif kernel_type == 'lanczos': 
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                
                if phase == 0.5:
                    di = np.abs(i + 0.5 - center) / factor  
                    dj = np.abs(j + 0.5 - center) / factor 
                else:
                    di = np.abs(i - center) / factor
                    dj = np.abs(j - center) / factor
                
                
                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                
                kernel[i - 1][j - 1] = val
                
                
    kernel /= kernel.sum()
    return kernel
    
    
#################

def extra_downsample(inputs,n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False,reuse=False,data_format='NHWC',name=None):

    assert phase in [0, 0.5], 'phase should be 0 or 0.5'
    
    if kernel_type == 'lanczos2':
        support = 2
        kernel_width = 4 * factor + 1
        kernel_type_ = 'lanczos'

    elif kernel_type == 'lanczos3':
        support = 3
        kernel_width = 6 * factor + 1
        kernel_type_ = 'lanczos'

    elif kernel_type == 'gauss12':
        kernel_width = 7
        sigma = 1/2
        kernel_type_ = 'gauss'

    elif kernel_type == 'gauss1sq2':
        kernel_width = 9
        sigma = 1./np.sqrt(2)
        kernel_type_ = 'gauss'

    elif kernel_type in ['lanczos', 'gauss', 'box']:
        kernel_type_ = kernel_type

    kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma).transpose(1,0)  #(kernel_width,kernel_width) or #(kernel_width-1,kernel_width-1)
    
    conv_weights = np.zeros([kernel.shape[0],kernel.shape[1],n_planes,n_planes]).astype(kernel.dtype)
        
    for i in range(n_planes):        
        conv_weights[:,:,i,i] = kernel

    conv_weights_init = tf.constant_initializer(conv_weights)    
    print(conv_weights.shape)
    if preserve_size:
        
        if kernel.shape[0] % 2 == 1:
            pad = int((kernel.shape[0] - 1) / 2.)
        else:
            pad = int((kernel.shape[0] - factor) / 2.)
                
        x = replication_pad2d(inputs,pad,data_format=data_format)
        
    else:
        x = inputs
        
    return tf.contrib.layers.conv2d(inputs=x,num_outputs=n_planes,kernel_size=kernel.shape,stride=factor,padding='valid',activation_fn=None,weights_initializer=conv_weights_init,biases_initializer=tf.zeros_initializer(),reuse=reuse,scope=name)

        
def torch_to_tf_conv2d(inputs,num_outputs,kernel_size,stride=1,bias=True,to_pad=0,pad_mode=None,weights_initializer=None,biases_initializer=None,scope='conv'):
    
    k = 1/(inputs.get_shape().as_list()[-1]*np.prod(kernel_size)) 
    distribution_limit = tf.cast(tf.sqrt(k),tf.float32)
    if weights_initializer is None:
        weights_initializer = tf.random_uniform_initializer(-distribution_limit,distribution_limit)
        
    if not bias:
        biases_initializer = None
        
    else:
        biases_initializer = tf.random_uniform_initializer(-distribution_limit,distribution_limit)        
        
    if to_pad != 0 and pad_mode is not None:
        if pad_mode != 'reflect':
            pad_mode = 'constant'
    
        paddings = tf.constant([[0,0],[to_pad,to_pad],[to_pad,to_pad],[0,0]])
        inputs = tf.pad(inputs,paddings,pad_mode)
        
    return tf.contrib.layers.conv2d(inputs=inputs,num_outputs=num_outputs,kernel_size=kernel_size,stride=stride,padding = 'valid',activation_fn=None,weights_initializer=weights_initializer,biases_initializer=biases_initializer,scope=scope)
    
    
def conv_layer(inputs,num_outputs,kernel_size,stride=1,bias=True,pad_mode='zero',downsample_mode='stride',scope='prior_conv'):
    """
    Args :
    inputs : A Tensor of shape (N,H,W,C)
    num_outputs : The number of channels of output
    kernel_size : A tuple or list of [kernel_height,kernel_width]
    """
    
    downsampler = None
    
    if stride != 1 and downsample_mode != 'stride':
    
        if downsample_mode == 'avg':            
            downsampler = 0
            
        elif downsample_mode == 'max':            
            downsampler = 1
            
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = 2
            
        stride = 1
    
    to_pad = int((kernel_size - 1) / 2)
    with tf.variable_scope(scope):
        
        convolver = torch_to_tf_conv2d(inputs,num_outputs,kernel_size,stride=stride,bias=bias,to_pad=to_pad,pad_mode=pad_mode,weights_initializer=None,biases_initializer=None,scope='conv')
        
        if downsampler in [0,1,2]:
            if downsample_mode == 'avg':
                return tf.nn.avg_pool(convolver,stride,stride,padding='valid')
        
            elif downsample_mode == 'max':
                return tf.nn.max_pool(convolver,stride,stride,padding='valid')            
                
            elif downsample_mode in ['lanczos2', 'lanczos3']:
                return extra_downsample(convolver,n_planes=num_outputs, factor=stride, kernel_type=downsample_mode, phase=0.5, kernel_width=None, support=None, sigma=None, preserve_size=True,reuse=False,name='downsample')

        else:
            return convolver
        
       
        
            


def skip_layer(inputs,num_outputs,filter_skip_size,bias,pad_mode='reflect',act_fn_type='leaky_relu',is_training=False,scope='skip'):
    
    with tf.variable_scope(scope):
        output1 = conv_layer(inputs,num_outputs,kernel_size=filter_skip_size,bias=bias,pad_mode=pad_mode)
        output2 = tf.contrib.layers.batch_norm(output1,scale=True,is_training=is_training)
        act_fn = activation_fns(act_fn_type=act_fn_type)
        return act_fn(output2)
        
       
def deeper_layer(inputs,num_outputs,kernel_size,stride=1,bias=True,pad_mode='zero',act_fn_type='leaky_relu',downsample_mode='stride',is_training=False,scope='deeper'):

    act_fn = activation_fns(act_fn_type=act_fn_type)
    with tf.variable_scope(scope):
        output1 = conv_layer(inputs,num_outputs,kernel_size=kernel_size,stride=2,bias=bias,pad_mode=pad_mode,downsample_mode=downsample_mode,scope='conv1')
        output2 = tf.contrib.layers.batch_norm(output1,scale=True,is_training=is_training)
        output3 = act_fn(output2)
        
        output4 = conv_layer(output3,num_outputs,kernel_size=kernel_size,bias=bias,pad_mode=pad_mode,downsample_mode=downsample_mode,scope='conv2')
        output5 = tf.contrib.layers.batch_norm(output4,scale=True,is_training=is_training)
        output6 = act_fn(output5)        
       
        return output6
            

def upsample_layer(inputs,scale_factor=2,upsample_mode='nearest',scope='upsample'):
    
    with tf.variable_scope(scope):
            
        upsample_class = tf.keras.layers.UpSampling2D(size=scale_factor,interpolation=upsample_mode)
        return upsample_class(inputs)
        
        

def add_last_layer(inputs,num_outputs,kernel_size,bias=True,pad_mode='zero',downsample_mode='stride',act_fn_type='leaky_relu',is_training=False,need1x1_up=True,scope=None):
    
    act_fn = activation_fns(act_fn_type=act_fn_type)
    
    with tf.variable_scope(scope):
        bn = tf.contrib.layers.batch_norm(inputs,scale=True,is_training=is_training)
        output1 = conv_layer(bn,num_outputs,kernel_size,stride=1,bias=bias,pad_mode=pad_mode,downsample_mode=downsample_mode,scope='prior_conv1')
        output2 = tf.contrib.layers.batch_norm(output1,scale=True,is_training=is_training)
        output = act_fn(output2)        
        
        if need1x1_up:
            conv_output = conv_layer(output,num_outputs,kernel_size=1,stride=1,bias=bias,pad_mode=pad_mode,downsample_mode=downsample_mode,scope='prior_conv2')
            bn = tf.contrib.layers.batch_norm(conv_output,scale=True,is_training=is_training)
            output = act_fn(bn)
            
        return output
        
        
def skip_net(inputs,num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, is_training=True,
        pad_mode='zero', upsample_mode='nearest', downsample_mode='stride', act_fn_type='leaky_relu', 
        need1x1_up=True):
        
        
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales


    
    last_scale = n_scales - 1 

    model_inputs = inputs    
    skip_outputs = []
    for i in range(last_scale):       

        skip = None       

        if num_channels_skip[i] != 0:
            skip_scope = 'skip_{}'.format(i+1)
            skip = skip_layer(model_inputs,num_channels_skip[i],filter_skip_size,bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope=skip_scope)

        skip_outputs.append(skip)

        deeper_scope = 'deeper_{}'.format(i+1)
        deeper = deeper_layer(model_inputs,num_channels_down[i],filter_size_down[i],stride=2,bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,downsample_mode=downsample_mode[i],is_training=is_training,scope=deeper_scope)
        
        model_inputs = deeper       
        
    skip = None
    if num_channels_skip[-1] != 0:
        skip_scope = 'skip_{}'.format(n_scales)
        skip = skip_layer(model_inputs,num_channels_skip[-1],filter_skip_size,bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope=skip_scope)

    skip_outputs.append(skip)
    
    skip_outputs_tmp = skip_outputs[:]
    for i in range(n_scales):      
                 
        if i == 0:
            upsample = model_inputs
            
        else:
            upsample_scope = 'upsample_{}'.format(i+1)
            upsample = upsample_layer(model_inputs,scale_factor=2,upsample_mode=upsample_mode[i],scope=upsample_scope)  
            
        output = upsample        

        if num_channels_skip[last_scale-i] != 0:
            assert skip_outputs_tmp[-1] != None
            output = tf.concat((skip_outputs_tmp[-1],upsample),axis=-1)
    
        skip_outputs_tmp.pop()        
    
        last_scope = 'last_{}'.format(i+1)
        output = add_last_layer(output,num_channels_up[i],filter_size_up[i],bias=need_bias,pad_mode=pad_mode,downsample_mode=downsample_mode[i],act_fn_type=act_fn_type,is_training=is_training,need1x1_up=need1x1_up,scope=last_scope)
        model_inputs = output
    
    prior_output = conv_layer(output,num_output_channels,kernel_size=1,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode=downsample_mode[i],scope='prior_conv')
        
    if need_sigmoid:
        return tf.nn.sigmoid(prior_output)
        
    else:
        return prior_output
        
        

####### Layers for Resnet

def get_block(inputs,num_outputs,act_fn_type='leaky_relu',is_training=True,scope='block'):
    
    act_fn = activation_fns(act_fn_type=act_fn_type)
    with tf.variable_scope(scope):
        conv1 = torch_to_tf_conv2d(inputs,num_outputs,kernel_size=3,stride=1,bias=False,to_pad=1,pad_mode='constant',scope='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1,scale=True,is_training=is_training)
        output1 = act_fn(bn1)
        
        conv2 = torch_to_tf_conv2d(output1,num_outputs,kernel_size=3,stride=1,bias=False,to_pad=1,pad_mode='constant',scope='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2,scale=True,is_training=is_training)
        
        return bn2
        
def residual(inputs,num_outputs,act_fn_type='leaky_relu',is_training=True,scope='residual'):
    act_fn = activation_fns(act_fn_type=act_fn_type)
    inputs_shape = inputs.get_shape().as_list()  #[b,h,w,c]

    with tf.variable_scope(scope):
        block_output = get_block(inputs,num_outputs,act_fn_type=act_fn_type,is_training=is_training,scope='block')
        block_shape = block_output.get_shape().as_list()  #[b,h,w,c]
        
        if inputs_shape[1] != block_shape[1] or inputs_shape[2] != block_shape[2]:
            diff1 = inputs_shape[1]-block_shape[1]
            diff2 = inputs_shape[2]-block_shape[2]
            
            inputs_ = inputs[:,diff1 /2:block_shape[1] + diff1 / 2, diff2 / 2:block_shape[2] + diff2 / 2,:]
            # (b,block_shape[1],block_shape[2],c)
        else:
            inputs_ = inputs
            # (b,block_shape[1],block_shape[2],c)
            
        return inputs_+block_output
        
        
def resnet(inputs,num_outputs,num_blocks,num_channels,need_residual=True,act_fn_type='leaky_relu',is_training=True,need_sigmoid=True,pad_mode='reflect',scope='resnet'):

    act_fn = activation_fns(act_fn_type=act_fn_type)
    
    with tf.variable_scope(scope):
    
        conv1 = conv_layer(inputs,num_channels,kernel_size=3,stride=1,bias=True,pad_mode=pad_mode,downsample_mode='stride',scope='conv1')
        output = act_fn(conv1)
        
        for i in range(num_blocks):
            residual_scope = 'residual_{}'.format(i+1)
            output = residual(output,num_channels,act_fn_type=act_fn_type,is_training=is_training,scope=residual_scope)            
            
        output = torch_to_tf_conv2d(output,num_channels,kernel_size=3,stride=1,bias=True,to_pad=1,pad_mode='constant',weights_initializer=None,biases_initializer=None,scope='conv2')
        output = tf.contrib.layers.batch_norm(output,scale=True,is_training=is_training)
        
        output = conv_layer(output,num_outputs,kernel_size=3,stride=1,bias=True,pad_mode='zero',downsample_mode='stride',scope='conv3')
        
        if need_sigmoid:
            return tf.nn.sigmoid(output)
            
        else:
            return output
            
            
####### unet layers

def unet_conv2(inputs,num_outputs,use_norm=None,need_bias=True,pad_mode=None,act_fn_type='relu',is_training=True,scope=None):

    act_fn = activation_fns(act_fn_type=act_fn_type)
    
    with tf.variable_scope(scope):        
        
        if use_norm == 'batch_norm':
            
            conv1 = conv_layer(inputs,num_outputs,kernel_size=3,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode='stride',scope='conv1')
            bn1 = tf.contrib.layers.batch_norm(conv1,scale=True,is_training=is_training)
            output1 = act_fn(bn1)
            
            conv2 = conv_layer(output1,num_outputs,kernel_size=3,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode='stride',scope='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2,scale=True,is_training=is_training)
            output2 = act_fn(bn2)            
            
            
        elif use_norm == 'instance_norm':
        
            conv1 = conv_layer(inputs,num_outputs,kernel_size=3,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode='stride',scope='conv1')
            bn1 = tf.contrib.layers.instance_norm(conv1)
            output1 = act_fn(bn1)
            # print('output1 : {}'.format(output1))
            
            conv2 = conv_layer(output1,num_outputs,kernel_size=3,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode='stride',scope='conv2')
            bn2 = tf.contrib.layers.instance_norm(conv2)
            output2 = act_fn(bn2)
            # print('output2 : {}'.format(output2))           
            
            
        if use_norm is None:
        
            conv1 = conv_layer(inputs,num_outputs,kernel_size=3,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode='stride',scope='conv1')            
            output1 = act_fn(conv1)
            
            conv2 = conv_layer(output1,num_outputs,kernel_size=3,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode='stride',scope='conv2')            
            output2 = act_fn(conv2)
                        
            
        return output2
            
            
def unet_down(inputs,num_outputs,use_norm=None,need_bias=True,pad_mode=None,act_fn_type='relu',is_training=True,scope='unet_down'):
    
    act_fn = activation_fns(act_fn_type=act_fn_type)
    # print('unet down inputs : {}'.format(inputs))
    with tf.variable_scope(scope): 
        output = tf.nn.max_pool(inputs,2,2,padding='VALID')     
        output = unet_conv2(output,num_outputs,use_norm=use_norm,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='conv')
        
        return output 
        

  
    
def torch_to_tf_convtranspose2d(inputs,num_outputs,kernel_size,strides,bias=True,weights_initializer=None,biases_initializer=None):
        
    k = 1/(inputs.get_shape().as_list()[-1]*np.prod(kernel_size)) 
    distribution_limit = tf.cast(tf.sqrt(k),tf.float32)
    if weights_initializer is None:
        weights_initializer = tf.random_uniform_initializer(-distribution_limit,distribution_limit)
        
    if not bias:
        biases_initializer = None
        
    else:
        biases_initializer = tf.random_uniform_initializer(-distribution_limit,distribution_limit)        
        
    # if to_pad != 0 and pad_mode is not None:
        # if pad_mode != 'reflect':
            # pad_mode = 'constant'    
        
        # paddings = tf.constant([[0,0],[to_pad,to_pad],[to_pad,to_pad],[0,0]])
        # inputs = tf.pad(inputs,paddings,pad_mode)
        
    transpose_layer = tf.keras.layers.Conv2DTranspose(filters=num_outputs,kernel_size=kernel_size,strides=strides,padding='same',use_bias=bias,kernel_initializer=weights_initializer,bias_initializer=biases_initializer)
        
    return transpose_layer(inputs)
        
def unet_up(inputs1,inputs2,num_outputs,need_bias,pad_mode,upsample_mode,act_fn_type='relu',same_num_filt=False,is_training=True,scope='unet_up'):

    inputs2_shape = inputs2.get_shape().as_list()
    
    with tf.variable_scope(scope):
        if upsample_mode == 'deconv':        
            
            # print('inputs1 : {}'.format(inputs1))
            # print('inputs2 : {}'.format(inputs2))
            up = torch_to_tf_convtranspose2d(inputs1,num_outputs,kernel_size=4,strides=2,bias=True,weights_initializer=None,biases_initializer=None)
            
            up_shape = up.get_shape().as_list()
            
            if up_shape[1] != inputs2_shape[1] or up_shape[2] != inputs2_shape[2]:
                diff1 = (inputs2_shape[1]-up_shape[1])//2
                diff2 = (inputs2_shape[2]-up_shape[2])//2
                inputs2_ = inputs2[:,diff1:diff1+up_shape[1],diff2:diff2+up_shape[2],:]
                
            else:
                inputs2_ = inputs2
                
            # print('unet up input1 : {}'.format(up))
            # print('unet up input2 : {}'.format(inputs2_))
            conv_input = tf.concat((up,inputs2_),axis=-1)
            conv = unet_conv2(conv_input,num_outputs*2,use_norm=None,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_conv2')
            
            
        elif upsample_mode in ['bilinear','nearest']:
            
            up = upsample_layer(inputs1,scale_factor=2,upsample_mode=upsample_mode,scope='upsample')
            up = conv_layer(up,num_outputs,kernel_size=3,stride=1,bias=need_bias,pad_mode=pad_mode,downsample_mode='stride',scope='up_conv')
            up_shape = up.get_shape().as_list()
            
            if up_shape[1] != inputs2_shape[1] or up_shape[2] != inputs2_shape[2]:
                diff1 = (inputs2_shape[1]-up_shape[1])//2
                diff2 = (inputs2_shape[2]-up_shape[2])//2
                inputs2_ = inputs2[:,diff1:diff1+up_shape[1],diff2:diff2+up_shape[2],:]
                
            else:
                inputs2_ = inputs2
                
            conv_input = tf.concat((up,inputs2_),axis=-1)
            conv = unet_conv2(conv_input,num_outputs*2,use_norm=None,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_conv2')
            
        return conv
        
        
def unet(inputs,num_input_channels=3,num_outputs=3,feature_scale=4,more_layers=0,concat_x=False,upsample_mode='deconv',pad_mode='zero',use_norm ='instance_norm',need_sigmoid=True,need_bias=True,act_fn_type='relu',is_training=True,scope='unet'):

    with tf.variable_scope(scope):
        
        filters = [2**i//feature_scale for i in range(6,11)]  #[16,32,64,128,256,512]
        
        downs = [inputs]
        
        for i in range(4+more_layers):
            downs.append(tf.nn.avg_pool(downs[-1],2,2,padding='VALID'))
        
        in64 = unet_conv2(inputs,filters[0] if not concat_x else filters[0] - num_input_channels,use_norm=use_norm,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='in64')
        
        if concat_x:
            in64 = tf.concat((in64,downs[0]),axis=-1)
        # print('in64 : {}'.format(in64))
        down1 = unet_down(in64,filters[1] if not concat_x else filters[1] - num_input_channels,use_norm=use_norm,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_down1')
        
        if concat_x:
            down1 = tf.concat((down1,downs[1]),axis=-1)
            
        # print('down1 : {}'.format(down1))
        down2 = unet_down(down1,filters[2] if not concat_x else filters[2] - num_input_channels,use_norm=use_norm,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_down2')
        
        if concat_x:
            down2 = tf.concat((down2,downs[2]),axis=-1)
            
        # print('down2 : {}'.format(down2))
        down3 = unet_down(down2,filters[3] if not concat_x else filters[3] - num_input_channels,use_norm=use_norm,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_down3')
        
        if concat_x:
            down3 = tf.concat((down3,downs[3]),axis=-1)
           
        # print('down3 : {}'.format(down3))
        down4 = unet_down(down3,filters[4] if not concat_x else filters[4] - num_input_channels,use_norm=use_norm,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_down4')
        
        if concat_x:
            down4 = tf.concat((down4,downs[4]),axis=-1)
            
        # print('down4 : {}'.format(down4))
            
        if more_layers > 0:
            prevs = [down4]
            
            for k in range(more_layers):
                out = unet_down(prevs[-1],filters[4] if not concat_x else filters[4] - num_input_channels,use_norm=use_norm,need_bias=need_bias,pad_mode=pad_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_down{}'.format(5+k))
                if concat_x:
                    out = tf.concat((out,downs[k+5]),axis=-1)
            
                prevs.append(out)
                
            last_idx = 5+more_layers
            up_ = unet_up(prevs[-1],prevs[-2],filters[4],need_bias,pad_mode,upsample_mode,act_fn_type=act_fn_type,same_num_filt=True,is_training=is_training,scope='unet_up_intermediate')
            for idx in range(more_layers-1):
                up_ = unet_up(up_,prevs[more_layers-idx-2],filters[4],need_bias,pad_mode,upsample_mode,act_fn_type=act_fn_type,same_num_filt=True,is_training=is_training,scope='unet_up{}'.format(last_idx+idx))
       

        else:
            up_ = down4
           
        # print('up_ : {}'.format(up_))
        up4 = unet_up(up_,down3,filters[3],need_bias,pad_mode,upsample_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_up4')
        up3 = unet_up(up4,down2,filters[2],need_bias,pad_mode,upsample_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_up3')
        up2 = unet_up(up3,down1,filters[1],need_bias,pad_mode,upsample_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_up2')
        up1 = unet_up(up2,in64,filters[0],need_bias,pad_mode,upsample_mode,act_fn_type=act_fn_type,is_training=is_training,scope='unet_up1')
        
        return conv_layer(up1,num_outputs,kernel_size=1,stride=1,bias=need_bias,pad_mode=pad_mode,scope='final')
        
        
############### texture net 


def texture_net_conv(inputs,num_outputs,kernel_size,stride=1,bias=True,pad_mode='constant',scope=None):
    to_pad = int((kernel_size-1)/2)    
    return torch_to_tf_conv2d(inputs,num_outputs,kernel_size,stride=stride,bias=True,to_pad=to_pad,pad_mode=pad_mode,weights_initializer=None,biases_initializer=None,scope=scope)
        
        

def gen_noise_layer(inputs,dim2,noise_type='uniform'):

    a = inputs.get_shape().as_list()    
    a = [tf.shape(inputs)[0],a[1],a[2],dim2]
    
    if noise_type == 'uniform':
        return tf.random.uniform(a)
        
    elif noise_type == 'normal':
        return tf.random.normal(a)
        
        
def texture_net_input_layer(inputs,num_outputs,num_blocks=3,is_training=True,act_fn_type='leaky_relu',use_upsample=True,upsample_mode='nearest',scope=None):

    act_fn = activation_fns(act_fn_type=act_fn_type)
    
    if not (isinstance(num_outputs, list) or isinstance(num_outputs, tuple)):
        num_outputs = [num_outputs]*num_blocks
    
    with tf.variable_scope(scope):
        
        for i in range(num_blocks):
            conv_scope = 'conv{}'.format(i+1)
            
            kernel_size = 3
            if i == num_blocks - 1:
                kernel_size = 1
                
            conv = texture_net_conv(inputs,num_outputs[i],kernel_size=kernel_size,stride=1,bias=True,pad_mode='constant',scope=conv_scope)
            bn = tf.contrib.layers.batch_norm(conv,scale=True,is_training=is_training)
            output = act_fn(bn)
            inputs = output
            
        if use_upsample:
            upsample = upsample_layer(inputs,scale_factor=2,upsample_mode=upsample_mode,scope='upsample')   
            return upsample
            # return tf.contrib.layers.batch_norm(upsample,scale=True,is_training=is_training)
            
        else:
            return inputs
            


    
def texture_net(inputs,conv_num=8,num_channels_inputs=3,ratios=[32, 16, 8, 4, 2, 1],fill_noise=False,pad_mode='constant',need_sigmoid=False,upsample_mode='nearest',is_training=True,scope='texture_net'):

    with tf.variable_scope(scope):
        
        for i in range(len(ratios)):
        
            j = i + 1
            
            avg_pool_scope = 'avg_pool{}'.format(i+1)            
            pooled = tf.nn.avg_pool2d(inputs,ratios[i],ratios[i],padding='VALID',name=avg_pool_scope)            
                        
            if fill_noise:
                pooled = gen_noise_layer(pooled,num_channels_inputs,noise_type='normal')
                
            # print('pooled : {}, {}'.format(i,pooled))
            layer_scope = 'output_{}'.format(i+1)
            num_outputs = conv_num * j 
            
            if i == 0:
                inputs1 = texture_net_input_layer(pooled,num_outputs,num_blocks=3,is_training=is_training,act_fn_type='leaky_relu',use_upsample=True,upsample_mode=upsample_mode,scope=layer_scope)
                
                # print('inputs1 : {}, {}'.format(i,inputs1))
            else:
            
                inputs2_scope = layer_scope+'inputs2'
                inputs2 = texture_net_input_layer(pooled,num_outputs,num_blocks=3,is_training=is_training,act_fn_type='leaky_relu',use_upsample=False,upsample_mode=upsample_mode,scope=inputs2_scope)
                inputs2 = tf.contrib.layers.batch_norm(inputs2,scale=True,is_training=is_training)
                
                # print('inputs2 : {}, {}'.format(i,inputs2))
                concat_ = tf.concat((inputs1,inputs2),axis=-1)
                
                inputs1 = texture_net_input_layer(concat_,num_outputs,num_blocks=3,is_training=is_training,act_fn_type='leaky_relu',use_upsample=False,upsample_mode=upsample_mode,scope=layer_scope)
                
                if i == len(ratios) - 1:
                    output = texture_net_conv(inputs1,num_outputs=3,kernel_size=1,stride=1,bias=True,pad_mode='constant',scope='final_conv')
                    
                else:
                    inputs1 = upsample_layer(inputs1,scale_factor=2,upsample_mode=upsample_mode,scope='upsample')  
                    
                    
                    
        return output