#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/8/17
"""

import sys
import cv2
import numpy as np
import scipy.misc
import gdal
import pandas as pd
import math

# Set some input variables
im_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/shapeAnalysis/wat_only_morph.tif'
prop_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/att_table_foo.csv'
cont_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours_foo.csv'
shapenames = ["obj","approx","hull","rect"]

triangle_shape = np.asarray([[[0,0]], [[4,0]], [[2,6]]])
# Function to read in image and save as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    return(file_handle.GetRasterBand(1).ReadAsArray())

# Function to calculate contour features
def cont_features(obj):
    # Needed for getting approximate poly
    epsilon = .05*cv2.arcLength(obj,True)

    # Basic shape dictionary
    shp_dict = {'obj':obj,'approx':cv2.approxPolyDP(obj,epsilon,True),'hull':cv2.convexHull(obj),'rect':cv2.boxPoints(cv2.minAreaRect(obj))}

    # Object moments
    shp_feats = cv2.moments(obj)

    # Match with triangle shape
    for match_meth in range(1,4):
        shp_feats['tri_match'+str(match_meth)] = cv2.matchShapes(obj,triangle_shape,match_meth,0)


        
    # Get areas and all that good stuff
    for shp in shapenames :
        shp_feats[shp + "_area"] = cv2.contourArea(shp_dict[shp])
        shp_feats[shp + "_perim"] = cv2.arcLength(shp_dict[shp],True)
        
    # Circle is separate because it's a little messier
    (circx,circy),circradius = cv2.minEnclosingCircle(obj)
    shp_feats["circ_rad"] = circradius

    # Return dictionary
    return(shp_feats)
    
# Funciton to caclulate derived features
def derived_features(fd):
    fd['circ_rsq'] = math.pow(fd['circ_rad'],2)
    fd['eq_diam'] = math.sqrt((4*fd['obj_area'])/math.pi)
    fd['solidity'] = fd['obj_area']/fd['hull_area']
    return(fd)


def main():
    # Read image
    wat_im = read_image(im_path)

    # Read in 
    # Get contours
    im2, contours, hierarchy = cv2.findContours(wat_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Create array for storing contours
    cont_df = pd.DataFrame(columns=["id","contour"])
    cont_df['contour'].astype(object)
    cont_id = 0
    for cnt in contours:
        feat_dict = cont_features(cnt)
        if feat_dict['obj_area']==0:
            continue
        feat_dict = derived_features(feat_dict)
        
        if 'prop_df' not in locals():
            colnames = ["id"] + feat_dict.keys()
            prop_df = pd.DataFrame(columns = colnames)
            
        prop_df.loc[cont_id,colnames] = [cont_id] + feat_dict.values()
        cont_df.loc[cont_id] = [cont_id] + [cnt.tolist()]

        
        
        cont_id+=1
        print(cont_id)
    prop_df.to_csv(prop_outpath)
    cont_df.to_csv(cont_outpath)
    
if __name__ == '__main__':
    main()
