#!/usr/bin/env python

import numpy as np
import pandas as pd

from ..res_io import read_write

# Given the points, find the indices in the image array
def get_train_indices(coords,geotrans):
    x_indices,y_indices = (((coords[:,0]-geotrans[0])/geotrans[1]).astype(int),
                           ((coords[:,1]-geotrans[3])/geotrans[5]).astype(int))
    yx_indices = np.stack([y_indices,x_indices],axis=1)
    return(yx_indices)

# Inputs: paths to csv of positive and negative training points, a labeled image
# Output: array of region IDs that are positive and negative
def training_ids(train_csv_path,labeled_image_path):
    labeled_image,geotrans = read_write.read_image(labeled_image_path)

    ysize,xsize = labeled_image.shape

    train_df = pd.read_csv(train_csv_path)
    pos_coords = train_df.loc[train_df['Class']==1].as_matrix(['X','Y'])
    neg_coords = train_df.loc[train_df['Class']==0].as_matrix(['X','Y'])
    
    pos_indices = get_train_indices(pos_coords,geotrans)
    neg_indices = get_train_indices(neg_coords,geotrans)

    # Eliminate any indices that are out of bounds
    pos_indices_inbounds = pos_indices[(pos_indices[:,0]>0) &
                                       (pos_indices[:,0]<ysize) &
                                       (pos_indices[:,1]>0) &
                                       (pos_indices[:,1]<xsize),]
    neg_indices_inbounds = neg_indices[(neg_indices[:,0]>0) &
                                       (neg_indices[:,0]<ysize) &
                                       (neg_indices[:,1]>0) &
                                       (neg_indices[:,1]<xsize),]
    
    pos_check = labeled_image[pos_indices_inbounds[:,0],pos_indices_inbounds[:,1]]
    pos_region_ids = pos_check[np.nonzero(pos_check)]
    
    neg_check = labeled_image[neg_indices_inbounds[:,0],neg_indices_inbounds[:,1]]
    neg_region_ids = neg_check[np.nonzero(neg_check)]
    
    return(pos_region_ids,neg_region_ids)
