#!/usr/bin/env python
## SCRIPT ##
"""
@authors: Kylen Solvik
Date Create: 3/8/17
- Builds attribute table for reservoir classification with shape descriptors
and INTENSITY values within contour
- Arguments:
1) wat_tif_path: Path to output from grow_shrink.py. \
   Image of opened/closed water objects.
2) intensity_tif_path: Path to INTENSITY file 
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
import os

from res_modules.add_features import calc_feats, find_training
from res_modules.res_io import read_write, split_recombine

#===============================================================================
# Get command line arguments
wat_tif = sys.argv[1]
intensity_tif = sys.argv[2]
tile_dir = sys.argv[3]
split = (sys.argv[4] == "True")
pos_training_csv = sys.argv[5]
neg_training_csv = sys.argv[6]
prop_csv_outpath = sys.argv[7]

tile_size_x = 8000
tile_size_y = 8000
overlap_size = 500

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
    if split:
        split_recombine.split_raster(wat_tif,tile_dir+"/water","water_",tile_size_x,tile_size_y,
                     overlap_size)
        print(intensity_tif)
        split_recombine.split_raster(intensity_tif,tile_dir+"/intensity","intensity_",tile_size_x,tile_size_y,
                     overlap_size)

    # Get tile_ids
    ds = gdal.Open(wat_tif)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    tile_ids = []
    for i in range(0, xsize, (tile_size_x-overlap_size)):
        for j in range(0, ysize, (tile_size_y-overlap_size)):
            tile_ids.append(str(i) + "_" + str(j)) 

    # Create output dirs
    if not os.path.exists(tile_dir+"/labeled"):
        os.makedirs(tile_dir+"/labeled")
    prop_csv_dir = os.path.dirname(prop_csv_outpath)
    if not os.path.exists(prop_csv_dir):
        os.makedirs(prop_csv_dir)
                
    # Calculate features for each tile
    create_csv = True
    for tile in tile_ids:
        print(tile)
        
        wat_im_path = tile_dir+"/water/water_"+tile+".tif"
        intensity_im_path = tile_dir+"/intensity/intensity_"+tile+".tif"
        labeled_out_path = tile_dir+"/labeled/labeled_"+tile+".tif"

        # Check if there are any water objects before calculating features
        wat_im,foo = read_write.read_image(wat_im_path)
        if (wat_im.max()>0):
            valid, feature_dataframe = calc_feats.shape_feats(wat_im_path,intensity_im_path,
                                                labeled_out_path,prop_list_get)
            if valid:
                # Find training examples
                pos_ids,neg_ids = find_training.training_ids(pos_training_csv,
                                                             neg_training_csv,
                                                             labeled_out_path)

                # Identify training examples in dataframe
                feature_dataframe.loc[feature_dataframe['id'].
                                      isin([tile + "-" + str(i) for i in pos_ids]),
                                      'class'] = 2
                feature_dataframe.loc[feature_dataframe['id'].
                                      isin([tile + "-" + str(i) for i in neg_ids]),
                                      'class'] = 1
        
                # Append to csv
                if create_csv:
                    feature_dataframe.to_csv(prop_csv_outpath, mode='w',
                                             header=True, index=False)
                    create_csv = False
                else: 
                    feature_dataframe.to_csv(prop_csv_outpath, mode='a',
                                             header=False, index=False)
        
    return()

if __name__ == '__main__':
    main()
