#!/usr/bin/env python

import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import gdal
from os import path
import pandas as pd

from read_write import *

# Fill out feature dict 
def get_feat_dict(i,plist,tile_id,plist_get):
    fdict = {}
    for get_prop in plist_get:
        prop_val = plist[i][get_prop]
        prop_type = type(prop_val)
        if prop_type is np.ndarray:
            prop_val_list = prop_val.flatten().tolist()
            for j in range(0,len(prop_val_list)):
                fdict[get_prop + str(j)] = prop_val_list[j]
        elif prop_type is tuple:
            prop_val_list = list(prop_val)
            for j in range(0,len(prop_val_list)):
                fdict[get_prop + str(j)] = prop_val_list[j]
        else:
            fdict[get_prop] = prop_val
    return(fdict)

# Find some extra intensity features based on OUTSIDE the region
def calc_intensity_feats(int_im,bbox,region):
    int_bbox = int_im[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    outsidereg = np.invert(region)
    if(np.any(outsidereg)):
        intensity_vals = int_bbox[outsidereg]
        outside_mean_int = np.mean(intensity_vals)
        outside_max_int = np.max(intensity_vals)
        outside_min_int = np.min(intensity_vals)
        outside_sd_int = np.std(intensity_vals)
    else:
        outside_mean_int = np.nan
        outside_max_int = np.nan
        outside_min_int = np.nan
        outside_sd_int = np.nan
    inside_sd_int = np.std(int_bbox[region])
    return([outside_mean_int,outside_max_int,outside_min_int,outside_sd_int,
            inside_sd_int])
    
    

# Main function to calculate all the features
def calc_shape_features(wat_im_path,intensity_im_path,labeled_out_path,plist_get):

    # Read Images
    wat_im,geotrans = read_image(wat_im_path)
    intensity_im,foo = read_image(intensity_im_path)

    # Get the ID of the current tile
    tile_id = '_'.join(path.splitext(path.basename(wat_im_path))[0].split("_")[-2:])

    # Calculate regionprops
    wat_clear = clear_border(wat_im)
    wat_labeled = label(wat_clear)
    plist = regionprops(wat_labeled, intensity_image = intensity_im)
    
    # Save water labeled tiff
    write_image(wat_labeled,wat_im_path,labeled_out_path,gdal.GDT_UInt16)
    
    # Construct feature df
    for i in range(0,len(plist)):
        
        feature_dict = get_feat_dict(i,plist,tile_id,plist_get)

        # Add extra features
        extra_int_feats = calc_intensity_feats(intensity_im,plist[i].bbox,
                                               plist[i].image)
        
        feature_dict.update({'out_mean_int':extra_int_feats[0],
                             'out_max_int':extra_int_feats[1],
                             'out_min_int':extra_int_feats[2],
                             'out_sd_int':extra_int_feats[3],
                             'in_sd_int':extra_int_feats[4]})
        
        colnames = ['id','class'] + feature_dict.keys()

        if i == 0:
            feature_df = pd.DataFrame(columns = colnames)
        
        feature_df.loc[i,colnames] = [tile_id + "-" + str(i+1),0] + feature_dict.values()

    return(feature_df)
