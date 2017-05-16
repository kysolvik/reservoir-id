#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/8/17
- Builds attribute table for reservoir classification with shape descriptors,
distance to plains, and NDWI values within contour. 
- Inputs
1) im_path: Path to output from grow_shrink.py. Image of opened/closed water objects.
2) ndwi_path: Path to NDWI file 
3) training_csv_path:
- Outputs: 
1) prop_outpath: Attribute table for machine learning
2) cont_outpath: csv containing contour IDs and shapes. Needed for plotting 
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
from feature_calc_funcs import *
import gc
#===============================================================================

# Test mode on means that it won't run for non-test set objects
full_run = True

# Set some input variables
lead_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/'
im_path = lead_path + 'intermediate/water_morph.tif'
ndwi_path = lead_path + 'inputs/ndwi_10m_goias.tif'
training_csv_path = lead_path + 'build_attribute_table/training_points/'

# Output paths
prop_outpath = lead_path + 'tables/att_table.csv'
cont_outpath = lead_path + 'tables/contours.csv'

# Some parameters for the calculating shape descriptors
triangle_shape = np.asarray([[[0,0]], [[4,0]], [[2,12]]])
area_cutoff = 500000
bounding_geos = ['obj','approx','hull','rect']

#===============================================================================

# Function to read in image and save as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    gt = file_handle.GetGeoTransform()
    return(file_handle.GetRasterBand(1).ReadAsArray(),gt)

#===============================================================================

def main():
    # Read images
    wat_im,geotrans = read_image(im_path)
    #ndwi_im, ndwi_geotrans = read_image(ndwi_path)
    ## Read in training points csv
    #res_csv = np.genfromtxt(training_csv_path + "all_res.csv",
    #                        delimiter=",",skip_header=1)
    #nonres_csv = np.genfromtxt(training_csv_path + "all_nonres.csv",
    #                           delimiter=",",skip_header=1)
    
    # Get contours
    im2, contours, hierarchy = cv2.findContours(wat_im,cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    print('Number of Objects:' + str(len(contours)))
    # Create array for storing contours
    cont_df = pd.DataFrame(columns=['id','contour'])
    cont_df['contour'].astype(object)
    cont_id = 0
    print_id = 0
    for cnt in contours:

        # Get class
        feat_class = 0 #set_train_class(cnt,res_csv,nonres_csv,geotrans)
        if (full_run or feat_class!=0):
            feat_dict = cont_features(cnt,bounding_geos,area_cutoff,triangle_shape)
            if feat_dict['obj_area']==0:
                continue
            elif feat_dict['obj_area']>area_cutoff:
                prop_df.loc[cont_id,['id','obj_area']] = [cont_id] \
                                                         + [feat_dict['obj_area']]
                print("Too Big!")
            else:
                # Calculate extra features
                feat_dict = derived_features(feat_dict)
#                (feat_dict['ndwi_min'],feat_dict['ndwi_max'],
#                 feat_dict['ndwi_mean']) = calc_nd_feats(cnt,ndwi_im)
                
                if 'prop_df' not in locals():
                    colnames = ['id','class'] + feat_dict.keys()
                    prop_df = pd.DataFrame(columns = colnames)
                    
                prop_df.loc[cont_id,colnames] = [cont_id,feat_class] + feat_dict.values()
                    
            cont_df.loc[cont_id] = [cont_id] + [cnt.tolist()]
                    
            cont_id+=1
            print_id+=1
            if print_id > 99 :
                print cont_id
                print_id=0
                        
    prop_df.to_csv(prop_outpath,index=False)
    cont_df.to_csv(cont_outpath,index=False)
    
if __name__ == '__main__':
    main()