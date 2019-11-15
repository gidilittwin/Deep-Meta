import json
import tensorflow as tf
import numpy as np
from src.utilities.moving_avg import MOV_AVG
from src.utilities import visualizer as VIS
from src.graphs import V1 as GRAPH
import os
import tfrecords_handler as TFH
from config import Config
MODE_TRAIN = 0
MODE_TEST  = 1
base_path='/Users/gidilittwin'

# Load config
config = Config.__init__(base_path=base_path)

if isinstance(config.categories, int):
    config.categories = [config.categories]

# Load model parameters from json file
MODEL_PARAMS = config.model_params_path
with open(MODEL_PARAMS, 'r') as f:
    model_params = json.load(f)
config.model_params = model_params    
    
 # Create directory if needed
directory = config.checkpoint_path + '/'+ config.experiment_name
if not os.path.exists(directory):
    os.makedirs(directory)

# Shapenet class names
with open('./classes.json', 'r') as f:
    classes2name = json.load(f)

#
grid_size_lr = config.grid_size
x            = np.linspace(-1, 1, grid_size_lr)
y            = np.linspace(-1, 1, grid_size_lr)
z            = np.linspace(-1, 1, grid_size_lr)
xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)



print('#############################################################################################')
print('###############################  '+config.experiment_name+'   ################################################')
print('#############################################################################################')


config.batch_size = 2




    

#%% Data iterators
train_iterator = TFH.iterator(config.path+'train/',
                              config.batch_size,
                              epochs=10000,
                              shuffle=True,
                              img_size=config.img_size[0],
                              im_per_obj=config.im_per_obj,
                              grid_size=config.grid_size,
                              num_samples=10000,
                              shuffle_size=config.shuffle_size,
                              categories = config.categories,
                              compression = config.compression)


test_iterator  = TFH.iterator(config.path+'test/',
                              1,
                              epochs=10000,
                              shuffle=False,
                              img_size=config.img_size[0],
                              im_per_obj=config.im_per_obj,
                              grid_size=config.grid_size,
                              num_samples=10000,
                              shuffle_size=config.shuffle_size,
                              categories = config.categories,
                              compression = config.compression)
    

idx_node          = tf.placeholder(tf.int32,shape=(), name='idx_node')  
level_set         = tf.placeholder(tf.float32,shape=(),   name='levelset')  
next_element      = train_iterator.get_next()
next_element_test = test_iterator.get_next()
next_batch        = TFH.process_batch_train(next_element,idx_node,config)
next_batch_test = TFH.process_batch_evaluate(next_element_test,idx_node,config)
with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.Variable(True, name='mode_node')



#%% Visualize data
if config.visualize:
    VIS.plot(train_iterator,next_element,next_batch,idx_node,config, mode='samples')


#%% Training graph and session
train_dict = GRAPH.build(next_batch,mode_node,level_set,config,batch_size=config.batch_size)
test_dict  = GRAPH.build(next_batch_test,mode_node,level_set,config,batch_size=config.test_size)
lr_node, train_op_cnn = GRAPH.optimizer(train_dict,config)
all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver    = tf.train.Saver(var_list=all_vars)
loader   = tf.train.Saver(var_list=all_vars)
session = tf.Session()
session.run(tf.initialize_all_variables())


#%% Train loop
loss_plot      = []
acc_plot       = []
iou_plot       = []
acc_plot_test  = []
iou_plot_test  = []
max_test_acc   = 0.
max_test_iou   = 0.
if config.finetune:
    loader.restore(session, directory+'/best_val-0')
    loss_plot     = np.load(directory+'/loss_values.npy')
    acc_plot      = np.load(directory+'/accuracy_values.npy')
    iou_plot      = np.load(directory+'/iou_values.npy')
    acc_plot_test = np.load(directory+'/accuracy_values_test.npy')
    iou_plot_test = np.load(directory+'/iou_values_test.npy')
    loss_plot     = np.split(loss_plot,loss_plot.shape[0])
    acc_plot      = np.split(acc_plot,acc_plot.shape[0])
    iou_plot      = np.split(iou_plot,iou_plot.shape[0])
    acc_plot_test = np.split(acc_plot_test,acc_plot_test.shape[0])
    iou_plot_test = np.split(iou_plot_test,iou_plot_test.shape[0])    
step           = 0
acc_mov        = MOV_AVG(300) # moving mean
loss_mov       = MOV_AVG(300) # moving mean
iou_mov        = MOV_AVG(300) # moving mean



session.run(mode_node.assign(True)) 
for epoch in range(1000000):
    session.run(train_iterator.initializer)
    while True:
        try:
            feed_dict = {lr_node            :config.learning_rate,
                         idx_node           :epoch%config.im_per_obj,
                         level_set          :config.levelset}     
            _, train_dict_ = session.run([train_op_cnn, train_dict],feed_dict=feed_dict) 
            acc_mov_avg  = acc_mov.push(train_dict_['accuracy'])
            loss_mov_avg = loss_mov.push(train_dict_['loss_class'])
            iou_mov_avg  = iou_mov.push(train_dict_['iou'])   

            if step % 10 == 0:
                print('Training: epoch: '+str(epoch)+' ,avg_accuracy: '+str(acc_mov_avg)+' ,avg_loss: '+str(loss_mov_avg)+' ,IOU: '+str(iou_mov_avg)+' ,max_test_IOU: '+str(max_test_iou))
            if step % config.plot_every == 0:
                acc_plot.append(np.expand_dims(np.array(acc_mov_avg),axis=-1))
                loss_plot.append(np.expand_dims(np.array(np.log(loss_mov_avg)),axis=-1))
                iou_plot.append(np.expand_dims(np.array(iou_mov_avg),axis=-1)) 
                np.save(directory+'/loss_values.npy',np.concatenate(loss_plot))
                np.save(directory+'/accuracy_values.npy',np.concatenate(acc_plot))
                np.save(directory+'/iou_values.npy',np.concatenate(iou_plot))
            if step % config.test_every == config.test_every -1:            
                acc_test, iou_test, classes_test, ids_test, ious_test = GRAPH.evaluate(test_iterator,
                                                                                       session,
                                                                                       mode_node,
                                                                                       config,
                                                                                       test_dict,
                                                                                       next_element_test,
                                                                                       lr_node,
                                                                                       idx_node,
                                                                                       level_set)
                acc_plot_test.append(np.expand_dims(np.array(acc_test),axis=-1))
                iou_plot_test.append(np.expand_dims(np.array(iou_test),axis=-1))            
                np.save(directory+'/accuracy_values_test.npy',np.concatenate(acc_plot_test))
                np.save(directory+'/iou_values_test.npy',np.concatenate(iou_plot_test))
                if iou_test>max_test_iou:
                    saver.save(session, directory+'/best_val', global_step=0)
                    max_test_iou = iou_test
                print('Testing:  max_test_accuracy: '+str(max_test_acc)+' ,max_test_IOU: '+str(max_test_iou))
            if step % config.save_every == config.save_every -1:  
                saver.save(session, directory+'/latest_train', global_step=step)
            step+=1                
        except tf.errors.OutOfRangeError:
            break
 
    




               

