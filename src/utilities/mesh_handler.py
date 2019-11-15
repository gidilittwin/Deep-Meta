import numpy as np

import plotly as py
import plotly.graph_objs as go


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


pl_deep = [[0.0, 'rgb(39, 26, 44)'],
           [0.1, 'rgb(53, 41, 74)'],
           [0.2, 'rgb(63, 57, 108)'],
           [0.3, 'rgb(64, 77, 139)'],
           [0.4, 'rgb(61, 99, 148)'],
           [0.5, 'rgb(65, 121, 153)'],
           [0.6, 'rgb(72, 142, 157)'],
           [0.7, 'rgb(80, 164, 162)'],
           [0.8, 'rgb(92, 185, 163)'],
           [0.9, 'rgb(121, 206, 162)'],
           [1.0, 'rgb(165, 222, 166)']]


def mesh_plot(obj,idx=0,type_='mesh'):
    

    traces = []
    vertices = obj[idx]['vertices']
    vertices_up = obj[idx]['vertices_up']
    faces    = obj[idx]['faces']

    
    # Grid limits
    traces.append(go.Scatter3d(x=[-1,-1,-1,-1,1,1,1,1],
                               y=[-1,-1,1,1,-1,-1,1,1],
                               z=[-1,1,-1,1,-1,1,-1,1],
                               mode='markers',
                               opacity=0.0,
                               marker=dict(size=0.0,opacity=0.0 ))   ) 
        
    # Mesh
    if type_=='mesh' or type_=='cubed':
        traces.append(go.Mesh3d(x=vertices[:,0],
                                y=vertices[:,1],
                                z=vertices[:,2],
#                                colorscale = [['0'  , 'rgba(20,29,67,0.6)'], 
#                                              ['0.5', 'rgba(51,255,255 ,0.6)'], 
#                                              ['1'  , 'rgba(255  ,191,0,0.6)']],                                           
                                intensity = vertices[:,2]*255,
                                colorscale='Viridis',
                                color='#FFB6C1',
                                opacity=1.0,
                                i=faces[:,0],
                                j=faces[:,1],
                                k=faces[:,2]))

    elif type_=='cloud':
        # Markers
        traces.append(go.Scatter3d(x=vertices_up[:,0],
                                    y=vertices_up[:,1],
                                    z=vertices_up[:,2],
                                    mode='markers',
                                    marker=dict(size=2,line=dict(color='rgba(0, 50, 255, 0.8)',width=1.5),opacity=0.8 ))   ) 
        

    elif type_=='cloud_up':
#    norm    = obj[idx]['norm']
#    norm = np.log(1+norm)
#    norm=norm/np.max(norm)        
        # Markers
#        norm = obj[idx]['norm']
        vectors = obj[idx]['jacobian']
        
        traces.append({'type':'cone',
                      'x':vertices_up[:,0],
                      'y':vertices_up[:,1],
                      'z':vertices_up[:,2],
                      'u':vectors[:,0],
                      'v':vectors[:,1],
                      'w':vectors[:,2],
#                      sizemode='scaled',
#                      sizeref=0.25, #this is the default value 
#                      showscale=True,
#                      colorscale=pl_deep, 
#                      colorbar=dict(thickness=20, ticklen=4), 
#                      anchor='tail'
                  })     
        
        traces.append(go.Scatter3d(x=vertices_up[:,0],
                                    y=vertices_up[:,1],
                                    z=vertices_up[:,2],
                                    mode='markers',
                                    marker=dict(size=2,line=dict(color='rgba(217, 217, 217, 0.8)',width=0.5),opacity=0.8 ))   ) 
        

    layout = dict(
        width=1200,
        height=1200,
        autosize=False,
        title='Mesh',
        scene=dict(
            xaxis=dict(range=[-1, 1],
                gridcolor='rgb(255, 255, 255)',
                showbackground=False,
                backgroundcolor='rgb(255, 255,255)'),
            yaxis=dict(range=[-1, 1],
                gridcolor='rgb(255, 255, 255)',
                showbackground=False,
                backgroundcolor='rgb(255, 255,255)'),
            zaxis=dict(range=[-1, 1],
                gridcolor='rgb(255, 255, 255)',
                showbackground=False,
                backgroundcolor='rgb(255,255,255)')
            ),)



    fig = dict(data=traces, layout=layout)
#    py.offline.plot(fig,auto_open=True, filename='/Users/gidilittwin/Desktop/plotly_'+type_+'_'+str(idx)+'.html')
    py.offline.plot(fig)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def double_mesh_plot(obj,idx=0,type_='mesh'):
    

    traces = []


    
    # Grid limits
    traces.append(go.Scatter3d(x=[-1,-1,-1,-1,1,1,1,1],
                               y=[-1,-1,1,1,-1,-1,1,1],
                               z=[-1,1,-1,1,-1,1,-1,1],
                               mode='markers',
                               opacity=0.0,
                               marker=dict(size=0.0,opacity=0.0 ))   ) 
        
    # Mesh
    vertices = obj[0]['vertices']
    faces    = obj[0]['faces']    
    traces.append(go.Mesh3d(x=vertices[:,0],
                            y=vertices[:,1],
                            z=vertices[:,2],
#                            colorscale = [['0'  , 'rgba(20,29,67,0.6)'], 
#                                          ['0.5', 'rgba(51,255,255 ,0.6)'], 
#                                          ['1'  , 'rgba(255  ,191,0,0.6)']],                                           
                            intensity = vertices[:,2]*255,
                            opacity=1.0,
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2]))

    vertices = obj[1]['vertices']
    faces    = obj[1]['faces']    
    traces.append(go.Mesh3d(x=vertices[:,0],
                            y=vertices[:,1],
                            z=vertices[:,2],
#                            colorscale = [['0'  , 'rgba(255,45,25 ,0.6)'], 
#                                          ['0.5', 'rgba(255,45,25 ,0.6)'], 
#                                          ['1'  , 'rgba(255,45,25 ,0.6)']],                                           
                            intensity = vertices[:,2]*110,
                            opacity=0.2,
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2]))
 

    layout = dict(
        width=1200,
        height=1200,
        autosize=False,
        title='Mesh',
        scene=dict(
            xaxis=dict(range=[-1.1, 1.1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(range=[-1.1, 1.1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(range=[-1.1, 1.1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(28,76,96)')
            ),)


    fig = dict(data=traces, layout=layout)
#    py.offline.plot(fig,auto_open=True, filename='/Users/gidilittwin/Desktop/plotly_'+type_+'_'+str(idx)+'.html')
    py.offline.plot(fig)
    
    
    

def cloud_plot(pointcloud,cloud_gt,reprojected,idx=0):
    
    traces = []
    pointcloud = (pointcloud+1)/2
    cloud_gt = (cloud_gt+1)/2
        
        
    # BONES GT
    for i in np.arange(0,len(skeleton21)):
        bones = skeleton21[i]
        for j in np.arange(1,bones.shape[0]):
            bone = cloud_gt[idx,bones[j-1:j+1],:]
            traces.append(go.Scatter3d(x=bone[:,0],
                                       y=bone[:,1],
                                       z=bone[:,2],
                                       mode='lines',
                                       line=dict(color='rgba(217, 217, 217, 0.14)',width=5)))
          
            
    # Markers
    traces.append(go.Scatter3d(x=pointcloud[idx,:,0],
                                y=pointcloud[idx,:,1],
                                z=pointcloud[idx,:,2],
                                mode='markers',
                                marker=dict(size=2,line=dict(color='rgba(217, 217, 217, 0.8)',width=0.5),opacity=0.8 ))   ) 
    
    # Projection
    colorscale=[['0'  , 'rgba(20,29,67,0.6)'], 
              ['0.5', 'rgba(51,255,255 ,0.6)'], 
              ['1'  , 'rgba(255  ,191,0,0.6)']]
    voxels_= reprojected
    vshape = voxels_.shape[1]
    voxels_= voxels_[idx,:,:,0]*255
    Y,X = np.meshgrid(np.linspace(0, 1, vshape),np.linspace(0, 1, vshape))
    traces.append(go.Surface(z=np.zeros(voxels_.shape),
                    x=X,
                    y=Y,
                    colorscale=colorscale,
                    showlegend=False,
                    showscale=False,
                    surfacecolor=voxels_))
        
    
    
    
    
    
    # Grid limits
    traces.append(go.Scatter3d(x=[0,0,0,0,1,1,1,1],
                               y=[0,0,1,1,0,0,1,1],
                               z=[0,1,0,1,0,1,0,1],
                               mode='markers',
                               marker=dict(size=0.0,opacity=1.0 ))   ) 
    
    
    layout = dict(
        width=1600,
        height=1200,
        autosize=False,
        title='PROCESSED_INPUT_DATA',
        scene=dict(
            xaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(28,76,96)')
            ),)


    fig = dict(data=traces, layout=layout)
    py.offline.plot(fig,auto_open=False, filename='/Users/gidilittwin/Desktop/plotly_input_data'+str(idx)+'.html')
#    py.offline.plot(fig)
    
    
        
def data_cloud_plot(pointcloud):
    
    traces = []
    pointcloud = (pointcloud+1)/2
              
            
    # Markers
    traces.append(go.Scatter3d(x=pointcloud[:,0],
                                y=pointcloud[:,1],
                                z=pointcloud[:,2],
                                mode='markers',
                                marker=dict(size=2,line=dict(color='rgba(217, 217, 217, 0.14)',width=0.5),opacity=0.8 ))   ) 
    # Grid limits
    traces.append(go.Scatter3d(x=[0,0,0,0,1,1,1,1],
                               y=[0,0,1,1,0,0,1,1],
                               z=[0,1,0,1,0,1,0,1],
                               mode='markers',
                               marker=dict(size=0.0,opacity=1.0 ))   ) 
    
    
    layout = dict(
        width=1200,
        height=800,
        autosize=False,
        title='Mesh',
        scene=dict(
            xaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(28,76,96)')
            ),)


    fig = dict(data=traces, layout=layout)
    py.offline.plot(fig)
        