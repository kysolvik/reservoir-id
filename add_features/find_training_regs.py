#!/usr/bin/env python

import sys
import os
import math
import numpy as np
import gdal
from os import path
from read_write import *

# # For testing
# pos_csv_path="/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/training_points/all_res.csv"
# neg_csv_path="/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/training_points/all_nonres.csv"
# labeled_image_path="./labeltest.tif"

# Given the points, find the indices in the image array
def get_train_indices(coords,geotrans):
    x_indices,y_indices = (((coords[:,0]-geotrans[0])/geotrans[1]).astype(int),
                           ((coords[:,1]-geotrans[3])/geotrans[5]).astype(int))
    yx_indices = np.stack([y_indices,x_indices],axis=1)
    return(yx_indices)

# Inputs: paths to csvs of positive and negative training points, a labeled image
# Output: array of region IDs that are positive and negative
def find_training_ids(pos_csv_path,neg_csv_path,labeled_image_path):
    labeled_image,geotrans = read_image(labeled_image_path)

    ysize,xsize = labeled_image.shape
    
    pos_coords = np.genfromtxt(pos_csv_path,delimiter=",",skip_header=1)[:,0:2]
    neg_coords = np.genfromtxt(neg_csv_path,delimiter=",",skip_header=1)[:,0:2]
    
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
