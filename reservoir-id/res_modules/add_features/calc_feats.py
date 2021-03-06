#!/usr/bin/env python

import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize
from os import path
import gdal
import pandas as pd
import math
from skimage.feature import hog
import time
from ..res_io import read_write

make_sq = True
bbox_grow_pixels= 50

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

# Expand the bounding box by a certain number of pixels, make square
def expand_bbox(bbox,grow_pix,make_sq_flag,im):
    side_lengths = [bbox[2]-bbox[0],bbox[3]-bbox[1]]
    im_dims = list(im.shape)[::-1]
    bbox_grow = [0,0]
    if make_sq_flag:
        max_side = side_lengths.index(max(side_lengths))
        min_side = abs(max_side-1)
        bbox_grow[max_side] = grow_pix
        bbox_grow[min_side] = int(bbox_grow[max_side] +
                                  round(side_lengths[max_side]-
                                        side_lengths[min_side])/2)
    else:
        bbox_grow = [50,50]
    new_bbox = (bbox[0] - bbox_grow[0],
                bbox[1] - bbox_grow[1],
                bbox[2] + bbox_grow[0],
                bbox[3] + bbox_grow[1])
    if any(b < 0 or b >= im_dims[0] for b in new_bbox):
        # If new bbox overlaps image edge, return false for first arg
        return(False,new_bbox)
    else:
        return(True,new_bbox)

# Find some extra intensity features based on OUTSIDE the region
def calc_intensity_feats(int_im,bbox,region):
    int_bbox = int_im[bbox[0]:bbox[2],
                      bbox[1]:bbox[3]]
    outsidereg = np.invert(region)
    int_feats_dict = {}
    if not np.any(outsidereg):
        print("Something wrong. No outside region")
    # For outside
    intensity_vals_out = int_bbox[outsidereg]
    int_feats_dict['out_mean_int'] = np.mean(intensity_vals_out)
    int_feats_dict['out_sd_int'] = np.std(intensity_vals_out)
    int_feats_dict['out_var_int'] = np.var(intensity_vals_out)
    for perc in range(0,101,5):
        int_feats_dict['out_'+str(perc)+'_int'] = (
            np.percentile(intensity_vals_out,perc))
    # For inside. Mean, min, and max already exist in region props
    intensity_vals_in = int_bbox[region]
    int_feats_dict['in_sd_int'] = np.std(intensity_vals_in)
    int_feats_dict['in_var_int'] = np.var(intensity_vals_in)
    for perc in range(5,96,5): 
        int_feats_dict['in_'+str(perc)+'_int'] = (
            np.percentile(intensity_vals_in,perc))
    return(int_feats_dict,int_bbox)

# Add all intensity pixels from resized bounding box
def get_pixel_feats(int_im,bbox):
    # Rescale
    int_bbox = int_im[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    resized_im = resize(int_bbox,(30,30),mode ='symmetric')    
    return(np.ndarray.flatten(resized_im))

# Add dervied features
def add_log_sqrt_sq(feature_dict):
    new_dict = feature_dict
    for key in feature_dict.keys():
        if np.isnan(feature_dict[key]):
            new_dict.update({"log"+key:np.nan})
            new_dict.update({"sq"+key:np.nan})
            new_dict.update({"sqrt"+key:np.nan})
        else:
            if feature_dict[key] == 0:
                new_dict.update({"log"+key:0})
            else:
                new_dict.update({"log"+key:math.log(abs(feature_dict[key]))})
            new_dict.update({"sq"+key:pow(feature_dict[key],2)})
            new_dict.update({"sqrt"+key:math.sqrt(abs(feature_dict[key]))})
    return(new_dict)

def calc_hog(int_bbox):
    resized_im = resize(int_bbox,(50,50),mode ='symmetric')
    fd = hog(resized_im, orientations=8, pixels_per_cell=(8, 8),
             cells_per_block=(1, 1), visualise=False,block_norm='L2-Hys')
    return(fd)

# Main function to calculate all the features
def shape_feats(wat_im_path,intensity_im_path,labeled_out_path,plist_get):

    # Read Images
    wat_im,geotrans = read_write.read_image(wat_im_path)
    intensity_im,foo = read_write.read_image(intensity_im_path)

    # Get the ID of the current tile
    tile_id = '_'.join(path.splitext(path.basename(wat_im_path))[0].split("_")[-2:])

    # Calculate regionprops
    wat_clear = clear_border(wat_im)
    wat_labeled = label(wat_clear)
    plist = regionprops(wat_labeled, intensity_image = intensity_im)
    
    # Save water labeled tiff
    read_write.write_image(wat_labeled,wat_im_path,labeled_out_path,gdal.GDT_UInt16)

    create_df_flag = True
    df_rownum = 0
    # Construct feature df
    for i in range(0,len(plist)):
        # Check if the expanded bbox is valid
        no_overlap,expanded_bbox = expand_bbox(plist[i].bbox,
                                               bbox_grow_pixels,make_sq,wat_im)
        if no_overlap:
#            ft_start = time.time()
            feature_dict = get_feat_dict(i,plist,tile_id,plist_get)

            ### Add extra features
            # Intensity mean, max, min, and quartiles inside and out of region
            expanded_bbox_reg = wat_labeled[expanded_bbox[0]:expanded_bbox[2]
                                       ,expanded_bbox[1]:expanded_bbox[3]] \
                                       == (i+1)
            extra_int_feats,int_bbox = calc_intensity_feats(intensity_im,
                                                            expanded_bbox,
                                                            expanded_bbox_reg)
            feature_dict.update(extra_int_feats)
            # # Pixel features from intensity bbox rescaled to 30,30
            # pix_val_array = get_pixel_feats(intensity_im,expanded_bbox)        
            # feature_dict.update({'pixval'+str(i):pix_val_array[i] for i in range(0,len(pix_val_array))})

            # # Histogram of gradient features
#            ht_start = time.time()
            hog_array = calc_hog(int_bbox)
            feature_dict.update({'hog'+str(i):hog_array[i] for i in range(0,len(hog_array))})
#            print(time.time() - ht_start)
            # # log, sqrt, and sq of all existing features
            # feature_dict = add_log_sqrt_sq(feature_dict)
            #print(feature_dict.keys())
            #print(feature_dict.values())
            colnames = ['id','class'] + feature_dict.keys()
            if create_df_flag:
                feature_df = pd.DataFrame(columns = colnames)
                create_df_flag = False

            feature_df.loc[df_rownum,colnames] = [tile_id + "-" + str(i+1),0] + feature_dict.values()
            df_rownum += 1
#            print(time.time() - ft_start)
            

    valid = not create_df_flag
    if valid:
        return(valid, feature_df)
    else:
        return(valid,"not valid")
