import tensorflow as tf
import numpy as np
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
from src.utilities.moving_avg import MOV_AVG



def g_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_shape(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function   

    
def f_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.regressor(current,args_)

    
    
def injection_wrapper(current,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.regressor(current,args_)

     
def build(next_batch,mode_node,level_set,config,batch_size=1):
    images                = next_batch['images'] 
    samples_sdf           = next_batch['samples_sdf']  
    samples_xyz           = next_batch['samples_xyz']
    evals_target          = {}
    evals_target['x']     = samples_xyz
    evals_target['y']     = samples_sdf
    evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
    g_weights             = f_wrapper(images,[mode_node,config])
    evals_function        = SF.sample_points_list(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = [batch_size,config.num_samples],samples=evals_target['x'] , use_samps=True)
    labels                = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(batch_size,-1)),0.0),tf.int64)
    logits                = tf.reshape(evals_function['y'],(batch_size,-1,1)) #- levelset
    logits_iou            = tf.concat((logits-level_set,-logits+level_set),axis=-1)
    logits_ce             = tf.concat((logits,-logits),axis=-1)

    predictions           = tf.nn.softmax(logits_iou)
    correct_prediction    = tf.equal(tf.argmax(predictions, 2), labels)
    accuracy              = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    err                   = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss_class            = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits_ce,name='cross-entropy'),axis=-1)
    loss                  = tf.reduce_mean(loss_class) #+ config.norm_loss_alpha*loss_sdf

    X                     = tf.cast(labels,tf.bool)
    Y                     = tf.cast(tf.argmax(predictions, 2),tf.bool)
    iou_image             = tf.reduce_sum(tf.cast(tf.logical_and(X,Y),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X,Y),tf.float32),axis=1)
    iou                   = tf.reduce_mean(iou_image)
    return {'loss':loss,'loss_class':tf.reduce_mean(loss_class),'accuracy':accuracy,'err':err,'iou':iou,'iou_image':iou_image}



def optimizer(train_dict,config):
    with tf.variable_scope('optimization_cnn',reuse=tf.AUTO_REUSE):
        cnn_vars      = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = '2d_cnn_model')
        lr_node       = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
        optimizer     = tf.train.AdamOptimizer(lr_node,beta1=config.beta1,beta2=0.999)
        grads         = optimizer.compute_gradients(train_dict['loss'],var_list=cnn_vars)
        global_step   = tf.train.get_or_create_global_step()
        clip_constant = 10
        g_v_rescaled  = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads]
        train_op_cnn  = optimizer.apply_gradients(g_v_rescaled, global_step=global_step)
    return lr_node, train_op_cnn




def evaluate(test_iterator, session, mode_node, config, test_dict, next_element_test, lr_node, idx_node, level_set):
    session.run(mode_node.assign(False)) 
    acc_mov_test  = MOV_AVG(3000000) # exact mean
    iou_mov_test  = MOV_AVG(3000000) # exact mean
    acc_mov_avg_test = 0
    iou_mov_avg_test = 0
    classes       = []
    ids           = []  
    ious          = []
    if config.fast_eval!=0:
        num_epochs = int(config.im_per_obj/config.test_size)
    else:
        num_epochs = 1
    for epoch_test in range(num_epochs):
        session.run(test_iterator.initializer)
        while True:
            try:
                feed_dict = {lr_node            :config.learning_rate,
                             idx_node           :epoch_test%config.im_per_obj,
                             level_set          :config.levelset}  
                accuracy_t_ ,iou_t_, iou_image_t, batch_ = session.run([test_dict['accuracy'], test_dict['iou'], test_dict['iou_image'], next_element_test],feed_dict=feed_dict) 
                acc_mov_avg_test = acc_mov_test.push(accuracy_t_)
                iou_mov_avg_test = iou_mov_test.push(iou_t_)
                classes.append(np.tile(batch_['classes'],(config.test_size,1)))
                ids.append(np.tile(batch_['ids'],(config.test_size,1)))
                ious.append(iou_image_t)   
            except tf.errors.OutOfRangeError:
                print('TEST::  epoch: '+str(epoch_test)+' ,avg_accuracy: '+str(acc_mov_avg_test)+' ,IOU: '+str(iou_mov_avg_test))
                break
    session.run(mode_node.assign(True)) 
    classes  = np.concatenate(classes,axis=0)[:,0]
    ious     = np.concatenate(ious,axis=0) 
    ids      = np.concatenate(ids,axis=0)[:,0]     
    return acc_mov_avg_test, iou_mov_avg_test, classes, ids, ious








