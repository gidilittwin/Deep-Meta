import numpy as np
import tensorflow as tf
from src.models.model_ops import cell1D, cell2D_res, CONV2D, cell1D_residual, cell2D_t
from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim


def mydropout(mode_node, x, prob):
    if prob!=1.0:
        return tf.cond(mode_node, lambda: tf.nn.dropout(x, prob), lambda: x)
    else:
        return x



#%% DETECTION
        

def volumetric_softmax(node,name):
    sums = tf.reduce_sum(tf.exp(node), axis=[1,2,3], keepdims=True)
    softmax = tf.divide(tf.exp(node),sums,name=name)
    return softmax
    
     

def resnet_config(example,args_):
    mode      = args_[0]
    config    = args_[1]
    in_node   = tf.image.resize_images(example,[137,137])
    ch_in     = in_node.get_shape().as_list()[-1]
    base_size = config.model_params['encoder']['base_size']
    with tf.variable_scope("Input"):
        params  = config.model_params['encoder']['input']
        conv0_w, conv0_b = CONV2D([params['k'],params['k'],ch_in,base_size])
        c0      = tf.nn.conv2d(in_node,conv0_w,strides=[1, params['stride'], params['stride'], 1],padding='SAME')
        current = tf.nn.bias_add(c0, conv0_b)
    with tf.variable_scope("Residuals"):
        for ii, layer in enumerate(config.model_params['encoder']['residuals']):
            if ii<config.bn_l0:
                BN = 0
            else:
                BN = config.batch_norm                
            current = cell2D_res(current, layer['k'], base_size*layer['s_in'],  base_size*layer['s_out'], mode, layer['stride'], 'r'+str(ii+1),use_bn=BN)#68
    
    featue_size_ = current.get_shape().as_list()
    current      = tf.nn.avg_pool(current,[1,featue_size_[1],featue_size_[2],1],[1,1,1,1],padding=config.padding)
    featue_size_ = current.get_shape().as_list()
    features     = tf.reshape(current,(-1,featue_size_[1]*featue_size_[2]*featue_size_[3]))
    return features
    




def sample_normal(shape, is_training=True, scope=None, deploy=False):
    # Note: is_training is tf.placeholder(tf.bool) type
    if deploy==False:
        return tf.cond(is_training,  
                    lambda: tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,name='std_norm'),
                    lambda: tf.zeros(shape,dtype=tf.float32,name='std_norm'))




def regressor(features,args_):
    mode    = args_[0]
    config  = args_[1]
    weights = []
    theta   = config.model_params['theta']
    featue_size  = tf.shape(features)
    with tf.variable_scope("fully",reuse=tf.AUTO_REUSE):

        for ii, layer in enumerate(config.model_params['decoder']):
            features = cell1D(features,layer['size'], mode, SCOPE='decode'+str(ii+1), with_act=layer['act'], with_bn=layer['batch_norm'])
        features = mydropout(mode, features, config.dropout)
        tf.add_to_collection('embeddings',features)
        features_size = features.get_shape().as_list()[-1]

        # branch out
        for ii in range(len(theta)):        
            if 'expand' in config.model_params.keys():
                current = features        
                for ll in range(len(config.model_params['expand'])):
                    factor  = config.model_params['expand'][ll]['factor']
                    current = cell1D(current,features_size*factor, mode, SCOPE='expand'+str(ii)+'_'+str(ll), with_act=True, with_bn=False)
            else:
                current = features
            layer_out = theta[ii]['w']
            layer_in  = theta[ii]['in']
            stdev    = 0.02
            ww = tf.reshape(cell1D(current,layer_in*layer_out, mode, SCOPE='w'+str(ii),stddev=stdev, with_act=False, with_bn=False),(featue_size[0],layer_in,layer_out) )
            bb = tf.reshape(cell1D(current,layer_out,          mode, SCOPE='b'+str(ii),stddev=stdev, with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            gg = 1.+ tf.reshape(cell1D(current,layer_out,          mode, SCOPE='g'+str(ii),stddev=stdev, with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            weights.append({'w':ww,'b':bb,'g':gg})
        tf.add_to_collection('weights',weights)
    return weights


