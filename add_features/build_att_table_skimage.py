#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/8/17
- Builds attribute table for reservoir classification with shape descriptors
and NDWI values within contour
- Arguments:
1) wat_tif_path: Path to output from grow_shrink.py. \
   Image of opened/closed water objects.
2) ndwi_tif_path: Path to NDWI file 
3) training_csv_dir:
4) prop_csv_outpath: Attribute table for machine learning
5) cont_csv_outpath: csv containing contour IDs and shapes. Needed for plotting 
classification results

"""

import sys
import cv2
import numpy as np
import scipy.misc
import gdal
import pandas as pd
import math
import time
import gc

from calc_region_props import *
from split_recombine_raster import *
from find_training_regs import *

#===============================================================================
# Test mode true means that it will only run for training set.
test_mode = True

# Get command line arguments
wat_tif = sys.argv[1]
ndwi_tif = sys.argv[2]
tile_dir = sys.argv[3]
pos_training_csv = sys.argv[4]
neg_training_csv = sys.argv[5]
prop_csv_outpath = sys.argv[6]

tile_size_x = 4000
tile_size_y = 4000
overlap_size = 250

prop_list_get = ['area','convex_area','eccentricity',
                 'equivalent_diameter','extent','inertia_tensor',
                 'inertia_tensor_eigvals','major_axis_length',
                 'max_intensity','mean_intensity','min_intensity',
                 'minor_axis_length','moments_normalized','moments_hu',
                 'orientation','perimeter','solidity',
                 'weighted_moments_normalized','weighted_moments_hu']
#===============================================================================

def main():

    # Split tifs
    tile_ids = split_raster(wat_tif,tile_dir+"/water","water_",tile_size_x,tile_size_y,
                 overlap_size)
    foo = split_raster(ndwi_tif,tile_dir+"/ndwi","ndwi_",tile_size_x,tile_size_y,
                 overlap_size)

    # Create output dirs
    if not os.path.exists(tile_dir+"/labeled"):
        os.makedirs(tile_dir+"/labeled")

    # Calculate features for each tile
    for tile in tile_ids:
        wat_im_path = tile_dir+"/water/water_"+tile+".tif"
        intensity_im_path = tile_dir+"/ndwi/ndwi_"+tile+".tif"
        labeled_out_path = tile_dir+"/labeled/labeled_"+tile+".tif"
        
        feature_dataframe = calc_shape_features(wat_im_path,intensity_im_path,
                                                labeled_out_path,prop_list_get)

        # Find training examples
        pos_ids,neg_ids = find_training_ids(pos_training_csv,neg_training_csv,
                                            labeled_out_path)

        # Identify training examples in dataframe
        feature_dataframe.loc[feature_dataframe['id'].
                              isin([tile + "-" + str(i) for i in pos_ids]) = 2
        feature_dataframe.loc[feature_dataframe['id'].
                              isin([tile + "-" + str(i) for i in neg_ids]) = 1
        
        # Append to csv
        if tile == tile_ids[0]:
            feature_dataframe.to_csv(prop_csv_outpath, mode='w', header=True)
        else: 
            feature_dataframe.to_csv(prop_csv_outpath, mode='a', header=False)
        
    return()
