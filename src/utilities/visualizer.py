import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from skimage import measure
import matplotlib.pyplot as plt


def plot(train_iterator,next_element,next_batch,idx_node,config, mode='voxels'):

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #session.run(mode_node.assign(False)) 
        sess.run(train_iterator.initializer)
        batch,batch_ = sess.run([next_element,next_batch],feed_dict={idx_node:0})
        idx=0
        if mode=='samples':
            vertices             = batch_['samples_xyz'][:,:,:]
            cubed = {'vertices':vertices[0,:,:],'faces':[],'vertices_up':vertices[0,:,:]}
            MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')

        elif mode=='voxels':
            psudo_sdf = batch['voxels'][idx,:,:,:]*1.0
            verts0, faces0, normals0, values0 = measure.marching_cubes_lewiner(psudo_sdf, 0.0)
            cubed0 = {'vertices':verts0/(config.grid_size-1)*2-1,'faces':faces0,'vertices_up':verts0/(config.grid_size-1)*2-1}
            MESHPLOT.mesh_plot([cubed0],idx=0,type_='mesh')

        elif mode=='vertices':
            vertices             = batch['vertices'][:,:,:]/(config.grid_size_v-1)*2-1
            cubed = {'vertices':vertices[idx,:,:],'faces':faces0,'vertices_up':vertices[idx,:,:]}
            MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')

        elif mode=='samples_in':
            vertices             = batch_['samples_xyz'][idx,:,:]
            vertices_on          = batch_['samples_sdf'][idx,:,:]<0.
            vertices              = vertices*vertices_on
            cubed = {'vertices':vertices,'faces':[],'vertices_up':vertices}
            MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')
        
        elif mode=='alpha':
            pic = batch_['images'][0,:,:,3]
            fig = plt.figure()
            plt.imshow(pic)

        elif mode=='rgb':
            pic = batch_['images'][0,:,:,0:3]
            fig = plt.figure()
            plt.imshow(pic)
