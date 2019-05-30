
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 05:21:13 2019

@author: nusrat
"""

import numpy as np

from dipy.viz import window, actor
from dipy.tracking.streamline import transform_streamlines
import vtk.util.colors as colors
from dipy.tracking import utils
from dipy.tracking.streamline import set_number_of_points
from sklearn.neighbors import KDTree
from dipy.tracking.distances import bundles_distances_mam
from sklearn import svm
import nibabel as nib
from joblib import Parallel, delayed

def show_tract(segmented_tract, color):
    """Visualization of the segmented tract.
    """ 
    affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
    bundle_native = transform_streamlines(segmented_tract, np.linalg.inv(affine))

    renderer = window.Renderer()
    stream_actor = actor.line(bundle_native, linewidth=0.1)
    bar = actor.scalar_bar()
    renderer.add(stream_actor)
    renderer.add(bar)
    window.show(renderer, size=(600, 600), reset_camera=False)          
    """Take a snapshot of the window and save it
    """
    window.record(renderer, out_path='bundle2.1.png', size=(600, 600))          

def load(filename):
    """Load tractogram from TRK file 
    """
    wholeTract= nib.streamlines.load(filename)  
    wholeTract = wholeTract.streamlines
    return  wholeTract    
def resample(streamlines, no_of_points):
    """Resample streamlines using 12 points and also flatten the streamlines
    """
    return np.array([set_number_of_points(s, no_of_points).ravel() for s in streamlines]) 
    
def build_kdtree(points, leafsize):
    """Build kdtree with resample streamlines 
    """
    return KDTree(points,leaf_size =leafsize)    
    
def kdtree_query(tract,kd_tree):
    """compute 1 NN using kdtree query and return the id of NN
    """
         
    dist_kdtree, ind_kdtree = kd_tree.query(tract, k=50)
    return np.hstack(ind_kdtree) 

def bundles_distances_mam_smarter_faster(A, B, n_jobs=-1, chunk_size=100):
    """Parallel version of bundles_distances_mam that also avoids
    computing distances twice.
    """
    lenA = len(A)
    chunks = chunker(A, chunk_size)
    if B is None:
        dm = np.empty((lenA, lenA), dtype=np.float32)
        dm[np.diag_indices(lenA)] = 0.0
        results = Parallel(n_jobs=-1)(delayed(bundles_distances_mam)(ss, A[i*chunk_size+1:]) for i, ss in enumerate(chunks))
        # Fill triu
        for i, res in enumerate(results):
            dm[(i*chunk_size):((i+1)*chunk_size), (i*chunk_size+1):] = res
            
        # Copy triu to trid:
        rows, cols = np.triu_indices(lenA, 1)
        dm[cols, rows] = dm[rows, cols]

    else:
        dm = np.vstack(Parallel(n_jobs=n_jobs)(delayed(bundles_distances_mam)(ss, B) for ss in chunks))

    return dm

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))
 
def create_train_data_set(train_subjectList,tract):
    
    T_filename_full_brain="/home/nusrat/Desktop/thesis_code/100307full1M.trk"
    wholeTractogram = load(T_filename_full_brain)
    
    train_data=[]
    for sub in train_subjectList:  
        print sub        
        T_filename="/home/nusrat/Desktop/thesis_code/"+sub+tract
        wholeTract = load (T_filename)       
        train_data=np.concatenate((train_data, wholeTract),axis=0) 
        
     
    ###################kdtree################# 
    print ("train data Shape") 
    print train_data.shape      
    resample_tractogram=resample(wholeTractogram,no_of_points=no_of_points)
    resample_tract=resample(train_data,no_of_points=no_of_points)
    
    
    kd_tree=build_kdtree (resample_tractogram, leafsize=leafsize)
    
    #kdtree query to retrive the NN id
    query_idx=kdtree_query(resample_tract, kd_tree)
    
    #extract the streamline from tractogram
    unique_query_idx= np.unique(np.array(query_idx))

    subsample_tract=wholeTractogram[ unique_query_idx]     
    
    wholeTract=np.array(wholeTract)
    x_train = bundles_distances_mam_smarter_faster(train_data, subsample_tract )
    
    return x_train,subsample_tract,train_data
    
   
      
if __name__ == '__main__':
    
    train_subjectList =["100307","124422","856766"]#,"161731","100307","245333","239944"]
    tract = "_af.left.trk"
    no_of_points=12    
    leafsize=10
        
   
    ################################ Train Data######################################

    print ("Preparing Train Data")
    x_train,subsample_tract,train_data= create_train_data_set(train_subjectList,tract)
    print x_train.shape
  
 
    ################labeling#######################    
#    siz=x_train.size
#    y=np.ones(siz)
    
    ###################### Test Data################################
    print ("Preparing Test Data")
    t_filename="/home/nusrat/Desktop/thesis_code/124422_af.left.trk"
     

    
    test_data=load(t_filename)    
    x_test =  bundles_distances_mam_smarter_faster(test_data,subsample_tract )   
       
    print ("test data Shape") 
    print x_test.shape 
    ##########################################
    
    ###########################one class SVM######################
    gamma_value = 0.0001
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma_value)
    clf.fit(x_train)
    ####################################################
    #x_train= np.array(x_train)
    x_pred_train=clf.predict(x_train.tolist())
    n_error_test = x_pred_train[x_pred_train==-1].size
    print('number of error for training =', n_error_test)
    
    x_pred_test=clf.predict(x_test.tolist())
    n_error_test = x_pred_test[x_pred_test==-1].size
    print('number of error for testing=',n_error_test)
    
    
    ###########################visualize tract######################
    test_data=np.array(test_data)
    segmented_tract_positive= test_data[np.where(x_pred_test==1)]
    segmented_tract_negative= test_data[np.where(x_pred_test==-1)]
    
    print("Show the tract")
    color_positive= colors.green
    color_negative=colors.red
    show_tract(segmented_tract_positive, color_positive,color_negative,segmented_tract_negative) 
    
    
  
