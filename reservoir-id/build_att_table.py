#!/usr/bin/env python
## SCRIPT ##
"""
@authors: Kylen Solvik
Date Create: 3/8/17
- Builds attribute table for reservoir classification
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
import multiprocessing as mp
import argparse

from res_modules.add_features import calc_feats, find_training
from res_modules.res_io import read_write, split_recombine

#===============================================================================
# Parse command line args
parser = argparse.ArgumentParser(description='Build attribute table for water objects.')
parser.add_argument('water_tif',
                    help='Path to black and white water/non-water tif',
                    type=str)
parser.add_argument('intensity_tif',
                    help='Path to intensity image. Usualy NDVI.',
                    type=str)
parser.add_argument('tile_dir',
                    help='Path to directory where computed tiles will be stored.',
                    type=str)
parser.add_argument('training_csv',
                    help='Path to training csv.',
                    type=str)
parser.add_argument('prop_csv_out',
                    help='Path for output csv with region properties.',
                    type=str)
parser.add_argument('--skip_split',
                    help='Skip splitting step. (Use tiles from a previous run.)',
                    action='store_true')
parser.add_argument('--path_prefix',
                    help='To be placed at beginnings of all other path args',
                    default='',
                    type=str)
args = parser.parse_args()

# Some hard coded variables
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

# Given tileid, run feature calculations
def tile_feat_calc(tile,q):
    
    wat_im_path = args.path_prefix + args.tile_dir + "/water/water_"+tile+".tif"
    intensity_im_path = args.path_prefix + args.tile_dir+"/intensity/intensity_"+tile+".tif"
    labeled_out_path = args.path_prefix + args.tile_dir+"/labeled/labeled_"+tile+".tif"

    # Check if there are any water objects before calculating features
    wat_im,foo = read_write.read_image(wat_im_path)
    if (wat_im.max()>0):
        valid, feature_dataframe = calc_feats.shape_feats(wat_im_path,intensity_im_path,
                                                          labeled_out_path,prop_list_get)
        if valid:
            # Find training examples
            pos_ids,neg_ids = find_training.training_ids(args.path_prefix + args.training_csv,
                                                         labeled_out_path)

            # Identify training examples in dataframe
            feature_dataframe.loc[feature_dataframe['id'].
                                  isin([tile + "-" + str(i) for i in pos_ids]),
                                  'class'] = 2
            feature_dataframe.loc[feature_dataframe['id'].
                                  isin([tile + "-" + str(i) for i in neg_ids]),
                                  'class'] = 1
            q.put(feature_dataframe)
    return()

# Take tasks from q and write to csv
def prop_csv_writer(q):
    while 1:
        m = q.get()
        if isinstance(m,basestring) and m == 'kill':
            print("done!")
            break
        # Append to csv
        if not os.path.isfile(args.path_prefix + args.prop_csv_out):
            m.to_csv(args.path_prefix + args.prop_csv_out, mode='w',
                      header=True, index=False)
        else:
            m.to_csv(args.path_prefix + args.prop_csv_out, mode='a',
                      header=False, index=False)
    return()
    
def main():
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() - 1)
    # Split tifs
    if split:
        split_recombine.split_raster(args.path_prefix + args.water_tif,
                                     args.path_prefix + args.tile_dir+"/water",
                                     "water_",tile_size_x,tile_size_y,
                                     overlap_size)
        print(args.path_prefix + args.intensity_tif)
        split_recombine.split_raster(args.path_prefix + args.intensity_tif,
                                     args.path_prefix + args.tile_dir+"/intensity","intensity_",
                                     tile_size_x,tile_size_y,
                                     overlap_size)

        print("Done with split")
    # Get tile_ids
    ds = gdal.Open(args.path_prefix + args.water_tif)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    tile_ids = []
    for i in range(0, xsize, (tile_size_x-overlap_size)):
        for j in range(0, ysize, (tile_size_y-overlap_size)):
            tile_ids.append(str(i) + "_" + str(j)) 

    # Create output dirs
    if not os.path.exists(args.path_prefix + args.tile_dir + "/labeled"):
        os.makedirs(args.path_prefix + args.tile_dir + "/labeled")
    prop_csv_dir = os.path.dirname(args.path_prefix + args.prop_csv_out)
    if not os.path.exists(prop_csv_dir):
        os.makedirs(prop_csv_dir)
                
    ### Calculate features for each tile
    # First start listener
    watcher = pool.apply_async(prop_csv_writer, (q,))
    # Start workers
    jobs = []
    for tile in tile_ids:
        job = pool.apply_async(tile_feat_calc,(tile,q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # Now that we are done, kill the listener
    q.put('kill')
    pool.close()
    return()

if __name__ == '__main__':
    main()
