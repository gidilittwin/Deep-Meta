
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



def skip_connection(SCOPE,l1,l2,k,M,N,stride):
    with tf.variable_scope(SCOPE+'_skip'):
        if M==N and stride==1:
            out = l1 + l2
        else:        
            conv_w, conv_b = CONV2D_I([k,k,M,N])
            reshape = tf.nn.conv2d(l1,conv_w,strides=[1, stride, stride, 1],padding='SAME')
            out = reshape + l2 
    return out        

def unravel_index(indices, shape, Type=tf.int32):
#    indices = tf.transpose(indices,(1,0))
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    shape = tf.cast(shape, tf.float32)
    strides = tf.cumprod(shape, reverse=True)
    strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
    strides = tf.cast(strides, Type)
    strides_shifted = tf.cast(strides_shifted, Type)
    def even():
        rem = indices - (indices // strides) * strides
        return rem // strides_shifted
    def odd():
        div = indices // strides_shifted
        return div - (div // strides) * strides
    rank = tf.rank(shape)
    return tf.cond(tf.equal(rank - (rank // 2) * 2, 0), even, odd)

def ravel_xy(bxy):
    shape = bxy.get_shape().as_list()
    x = tf.slice(bxy,[0,0],[-1,1])
    y = tf.slice(bxy,[0,1],[-1,1])
    return y + shape[1]*x 

def ravel_index(bxyz, shape):
    b = tf.slice(bxyz,[0,0],[-1,1])
    x = tf.slice(bxyz,[0,1],[-1,1])
    y = tf.slice(bxyz,[0,2],[-1,1])
    z = tf.slice(bxyz,[0,3],[-1,1])
    return z + shape[3]*y + (shape[2] * shape[1])*x + (shape[3] * shape[2] * shape[1])*b

def ravel_index_bxy(bxyz, shape):
    b = tf.slice(bxyz,[0,0],[-1,1])
    x = tf.slice(bxyz,[0,1],[-1,1])
    y = tf.slice(bxyz,[0,2],[-1,1])
    return y + (shape[2])*x + (shape[2] * shape[1])*b


def BatchNorm(inputT, is_training=True, scope=None, deploy=False):
    # Note: is_training is tf.placeholder(tf.bool) type
    if deploy==False:
        return tf.cond(is_training,  
                    lambda: batch_norm(inputT, is_training=True,  
                                       center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),  
                    lambda: batch_norm(inputT, is_training=False,  
                                       center=True, scale=True, decay=0.9, updates_collections=None, scope=scope, reuse = True))  
    else:
        return batch_norm(inputT, is_training=False,center=True, scale=True, decay=0.9, updates_collections=None, scope=scope, reuse = False)  


def lrelu(x, leak=0.2, name="LRelU"):
   with tf.variable_scope(name):
       return tf.maximum(x, leak*x)        

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, bn = True):
   shape = input_.get_shape().as_list()
   with tf.variable_scope(scope or "Linear"):
       matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
       bias = tf.get_variable("bias", [output_size],
           initializer=tf.constant_initializer(bias_start))
       output = tf.matmul(input_, matrix) + bias
       if with_w:
           return output, matrix, bias
       else:
           return output


def CONV2D(shape,learn_bias=True):
   conv_weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=np.sqrt(2)/np.sqrt(shape[0]*shape[1]*shape[2])))
   if learn_bias:
       conv_biases = tf.get_variable('biases',[shape[-1]], initializer=tf.constant_initializer(0.0))
   else:
       conv_biases = None
   tf.add_to_collection('l2_res',(tf.nn.l2_loss(conv_weights)/np.prod(shape)))
   return conv_weights, conv_biases



def CONV2D_I(shape):
   Ident = np.zeros(shape)
#   kernel = np.array([[0.0625, 0.125 , 0.0625],
#                   [0.125 , 0.25  , 0.125 ],
#                   [0.0625, 0.125 , 0.0625]])
   for i in range(shape[3]):
       Ident[0,0,i % shape[2],i] = 1.
   conv_weights = tf.get_variable(name='weights', initializer = Ident.astype('float32'),trainable = False, dtype='float32')
   conv_biases = tf.get_variable(name='biases', shape = [shape[-1]], initializer=tf.constant_initializer(0.0), trainable=False)
   return conv_weights, conv_biases 
    
def cell(in_node, k, M, N, mode_node, var_list, SCOPE, downsample=False):
    # Conv 1
    with tf.variable_scope(SCOPE+'1') as scope:
        conv1_w, conv1_b = CONV2D([k,k,M,N])
        if downsample==False:
            conv1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        else:
            conv1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 2, 2, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b) 
        conv1 = BatchNorm(conv1,mode_node,scope)
        relu1 = tf.nn.relu(conv1)
    # Conv 2
    with tf.variable_scope(SCOPE+'2') as scope:
        conv2_w, conv2_b = CONV2D([k,k,N,N])
        conv2 = tf.nn.conv2d(relu1,conv2_w,strides=[1, 1, 1, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b) 
        conv2 = BatchNorm(conv2,mode_node,scope)
    # Identity connection
    with tf.variable_scope(SCOPE+'3') as scope:
        if downsample==False:
            out = conv2 + in_node
        else:
            conv3_w, conv3_b = CONV2D_I([3,3,M,N])
            reshape = tf.nn.conv2d(in_node,conv3_w,strides=[1, 2, 2, 1],padding='SAME')
            out = reshape + conv2 
        out = tf.nn.relu(out)
    return out

    
  
       
def cell2D(in_node, k1, k2, M, N, mode_node, stride, SCOPE, padding='SAME', bn=True, act=True):
    with tf.variable_scope(SCOPE) as scope:
        conv1_w, conv1_b = CONV2D([k1,k2,M,N])
        conv1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, stride, stride, 1],padding=padding)
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        if bn==True:
            conv1 = BatchNorm(conv1,mode_node,scope)
        if act==True:
            conv1 = tf.nn.relu(conv1)
        print(conv1.get_shape())        
        return conv1

def cell2D_t(in_node, k1, k2, M, N, mode_node, stride, SCOPE, padding='SAME', bn=True, act=True):
    with tf.variable_scope(SCOPE) as scope:
        input_shape = in_node.get_shape().as_list()
        conv0_w, conv0_b = CONV2D([k1,k2,M,N])
        conv0_w = tf.transpose(conv0_w,(0,1,3,2))
        current = tf.nn.conv2d_transpose(in_node,
                                conv0_w,
                                [input_shape[0],input_shape[1]*2,input_shape[2]*2,N],
                                strides = [1,2,2,1]) +conv0_b
        if bn==True:
            current = BatchNorm(current,mode_node,scope)
        if act==True:
            current = tf.nn.relu(current)
        print(current.get_shape())        
        return current

        
def cell1D(in_node,output_size, mode_node, SCOPE=None, stddev=0.02, bias_start=0.0, with_act=True, with_bn=True,act_type=lrelu):
    with tf.variable_scope(SCOPE,reuse=tf.AUTO_REUSE) as scope:
        shape = in_node.get_shape().as_list()
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
           initializer=tf.constant_initializer(bias_start))
        tf.add_to_collection('l2_res',(tf.nn.l2_loss(matrix)))
        tf.add_to_collection('l2_res',(tf.nn.l2_loss(bias)))
        output = tf.matmul(in_node, matrix) + bias
        if with_bn:
            output = BatchNorm(output,mode_node,scope)
        if with_act:
            output = act_type(output)
        print(output.get_shape())        
        return output

def cell1D_residual(in_,width, mode_node, relu0=True, SCOPE=None):
    with tf.variable_scope(SCOPE,reuse=tf.AUTO_REUSE):
        shape_ = in_.get_shape().as_list()
        if relu0:
            c1 = tf.nn.relu(in_)
        else:
            c1 = in_
        c2 = cell1D(c1,width, mode_node, SCOPE='l1', with_act=False, with_bn=False, stddev=np.sqrt(2)/np.sqrt(shape_[-1]))
        c2 = tf.nn.relu(c2)
        c3 = cell1D(c2,width, mode_node, SCOPE='l2', with_act=False, with_bn=False, stddev=np.sqrt(2)/np.sqrt(width))
        if width==shape_[-1]:
            return in_+c3
        elif width==2*shape_[-1]:
            return tf.concat((in_,in_),axis=1)+c3
        


def cell2D_res(in_node, k, M, N, mode_node, stride, SCOPE,use_bn=True):
    with tf.variable_scope(SCOPE+'_1') as scope:
        if use_bn:
            batch1 = BatchNorm(in_node,mode_node,scope)
        else:
            batch1 = in_node
        relu1 = tf.nn.relu(batch1)
        conv1_w, conv1_b = CONV2D([k,k,M,N])
        conv1 = tf.nn.conv2d(relu1,conv1_w,strides=[1, stride, stride, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        print(conv1.get_shape())
    with tf.variable_scope(SCOPE+'_2') as scope:
        if use_bn:
            batch2 = BatchNorm(conv1,mode_node,scope)
        else:
            batch2 = conv1            
        relu2 = tf.nn.relu(batch2)
        conv2_w, conv2_b = CONV2D([k,k,N,N])
        conv2 = tf.nn.conv2d(relu2,conv2_w,strides=[1, 1, 1, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b)
        print(conv2.get_shape())         
    with tf.variable_scope(SCOPE+'_skip') as scope:
        if M==N and stride==1:
            out = conv2 + in_node
        else:        
            conv3_w, conv3_b = CONV2D_I([1,1,M,N])
            reshape = tf.nn.conv2d(in_node,conv3_w,strides=[1, stride, stride, 1],padding='SAME')
            out = reshape + conv2 
    return out

  




