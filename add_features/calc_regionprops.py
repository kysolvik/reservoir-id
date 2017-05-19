
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

# Read in tif as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    gt = file_handle.GetGeoTransform()
    return(file_handle.GetRasterBand(1).ReadAsArray(),gt)

# Function for writing out image with proper projetion
def write_image(im,inpath,outpath):
    
    # Input
    source_ds = gdal.Open(inpath)
    source_band = source_ds.GetRasterBand(1)
    #x_min, x_max, y_min, y_max = source_band.GetExtent()
    
    # Destination
    dst_filename = outpath
    y_pixels, x_pixels = im.shape  # number of pixels in x
    driver = gdal.GetDriverByName('GTiff')
    outds = driver.Create(dst_filename,x_pixels, y_pixels, 1,gdal.GDT_Byte)
    outds.GetRasterBand(1).WriteArray(im)
    
    # Add GeoTranform and Projection
    geotrans=source_ds.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=source_ds.GetProjection() #you can get from a exsited tif or import
    outds.SetGeoTransform(geotrans)
    outds.SetProjection(proj)
    outds.FlushCache()
    outds=None
                                                                                                                                            return()

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

def get_train_points(res_points,nonres_points,geotrans):
    x=int((pos[0] - geotrans[0])/geotrans[1])
    y=int((pos[1] - geotrans[3])/geotrans[5])
    return(x,y)

# Finds which labeled regions have manually been classified as reservoirs/nonreservoirs
# Inputs: paths to csvs of positive and negative training points, a labeled image
# Output: array of region IDs that are positive and negative
def find_train_regions(res_csv,nonres_csv,labeled_image_path):
    labeled_image,geotrans = read_image(labaled_image_path)

    
    
    res_point_array,nonres_point_array = get_train_points(res_points,
                                                          nonres_points,
                                                          geotrans)
    res_check = labeled_image[res_point_array]
    res_regions_ids = res_check[np.nonzero(res_check)]
    
    nonres_check = labeled_image[nonres_point_array]
    nonres_regions_ids = nonres_check[np.nonzero(nonres_check)]

    return(res_region_ids,nonres_region_ids)
    
    

# Main function to calculate all the features
def calc_shape_features(wat_im_path,intensity_im_path,plist_get):

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
    write_image(wat_labeled,wat_im,foopath)
    
    # Construct feature df
    for i in range(0,len(plist)):
        
        feature_dict = get_feat_dict(i,plist,tile_id,plist_get)

        if 'feature_df' not in locals():
            colnames =  ['id'] + feature_dict.keys()
            feature_df = pd.DataFrame(columns = colnames)
        
        feature_df.loc[i,colnames] = [tile_id + "-" + str(i)] + feature_dict.values()
    prop_csv_outpath="/Users/ksolvik/Documents/foo.csv"

    return(feature_df)
