#!/usr/bin/env python

import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize
from os import path
import gdal
import pandas as pd

from ..io import read_write

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
    int_bbox = int_im[bbox[0]:bbox[2],
                      bbox[1]:bbox[3]]
    outsidereg = np.invert(region)
    if(np.any(outsidereg)):
        intensity_vals_out = int_bbox[outsidereg]
        out_mean_int = np.mean(intensity_vals_out)
        out_max_int = np.max(intensity_vals_out)
        out_min_int = np.min(intensity_vals_out)
        out_sd_int = np.std(intensity_vals_out)
        out_25th_int = np.percentile(intensity_vals_out,25)
        out_median_int = np.percentile(intensity_vals_out,50)
        out_75th_int = np.percentile(intensity_vals_out,75)    
    else:
        out_mean_int = np.nan
        out_max_int = np.nan
        out_min_int = np.nan
        out_sd_int = np.nan
        out_25th_int = np.nan
        out_median_int = np.nan
        out_75th_int = np.nan
    intensity_vals_in = int_bbox[region]
    in_sd_int = np.std(intensity_vals_in)
    in_25th_int = np.percentile(intensity_vals_in,25)
    in_median_int = np.percentile(intensity_vals_in,50)
    in_75th_int = np.percentile(intensity_vals_in,75)
    return([out_mean_int,out_max_int,out_min_int,out_sd_int,out_25th_int,
            out_median_int,out_75th_int,in_sd_int,in_25th_int,in_median_int,
            in_75th_int])
    
# Add intensity
def get_pixel_feats(int_im,bbox):
    # Grow bounding box by 25% in each direction
    y_grow = int(round((bbox[2]-bbox[0])*.25))
    x_grow = int(round((bbox[3]-bbox[1])*.25))
    y_max,x_max = int_im.shape
    new_bbox = (max(bbox[0] - y_grow,1),max(bbox[1] - x_grow,1),
            min(bbox[2] + y_grow,y_max),min(bbox[3] + x_grow,x_max))
    # Rescale
    int_expanded_bbox = int_im[new_bbox[0]:new_bbox[2],new_bbox[1]:new_bbox[3]]
    resized_im = resize(int_expanded_bbox,(30,30),mode ='symmetric')
    return(np.ndarray.flatten(resized_im))
    
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
                             'out_25th_int':extra_int_feats[4],
                             'out_median_int':extra_int_feats[5],
                             'out_75th_int':extra_int_feats[6],
                             'in_sd_int':extra_int_feats[7],
                             'in_25th_int':extra_int_feats[8],
                             'in_median_int':extra_int_feats[9],
                             'in_75th_int':extra_int_feats[10]})
        # # Pixel features from intensity bbox rescaled to 50,50
        # pix_val_array = get_pixel_feats(intensity_im,plist[i].bbox)        
        # feature_dict.update({'pixval'+str(i):pix_val_array[i] for i in range(0,len(pix_val_array))})
 
        colnames = ['id','class'] + feature_dict.keys()

        if i == 0:
            feature_df = pd.DataFrame(columns = colnames)
        
        feature_df.loc[i,colnames] = [tile_id + "-" + str(i+1),0] + feature_dict.values()

    return(feature_df)
