#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/8/17
- Builds attribute table for reservoir classification with shape descriptors
and NDWI values within contour
- Arguments:
1) wat_tif_path: Path to output from grow_shrink.py. Image of opened/closed water objects.
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
from feature_calc_funcs import *
import gc

#===============================================================================

# Test mode true means that it will only run for training set.
test_mode = True


# Get arguments
wat_tif_path = sys.argv[1]
ndwi_tif_path = sys.argv[2]
training_csv_dir = sys.argv[3]
prop_csv_outpath = sys.argv[4]
cont_csv_outpath = sys.argv[5]

# Some parameters for the calculating shape descriptors
triangle_shape = np.asarray([[[0,0]], [[4,0]], [[2,12]]])
area_cutoff = 500000
bounding_geos = ['obj','approx','hull','rect']

#===============================================================================

def main():

    # Read images
    wat_im,geotrans = read_image(wat_tif_path)
    ndwi_im, ndwi_geotrans = read_image(ndwi_tif_path)

    # Read in training points csv
    res_csv = np.genfromtxt(training_csv_dir + "all_res.csv",
                            delimiter=",",skip_header=1)
    nonres_csv = np.genfromtxt(training_csv_dir + "all_nonres.csv",
                               delimiter=",",skip_header=1)
    
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
        feat_class = set_train_class(cnt,res_csv,nonres_csv,geotrans)
        if ( not test_mode  or feat_class!=0):
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
                (feat_dict['ndwi_min'],feat_dict['ndwi_max'],
                 feat_dict['ndwi_mean']) = calc_nd_feats(cnt,ndwi_im)
                
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
                        
    prop_df.to_csv(prop_csv_outpath,index=False)
    cont_df.to_csv(cont_csv_outpath,index=False)
    
if __name__ == '__main__':
    main()
