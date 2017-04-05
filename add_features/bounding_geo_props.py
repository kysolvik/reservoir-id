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

# Set some input variables
im_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/wat_only_clip_point.tif'
prop_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/att_table.csv'
cont_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours.csv'
shapenames = ["obj","approx","hull","rect"]

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

    # Get areas and all that good stuff
    for shp in shapenames :
        shp_feats[shp + "_area"] = cv2.contourArea(shp_dict[shp])
        shp_feats[shp + "_perim"] = cv2.arcLength(shp_dict[shp],True)
        
    # Circle is separate because it's a little messier
    (circx,circy),circradius = cv2.minEnclosingCircle(obj)
    shp_feats["circ_radius"] = circradius

    # Return dictionary
    return(shp_feats)
    

def main():
    # Read image
    wat_im = read_image(im_path)

    # Get contours
    im2, contours, hierarchy = cv2.findContours(wat_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Create array for storing contours
    cont_df = pd.DataFrame(columns=["id","contour"])
    cont_df['contour'].astype(object)
    cont_id = 0
    for cnt in contours:
        feat_dict = cont_features(cnt)

        if 'prop_df' not in locals():
            colnames = ["id"] + feat_dict.keys()
            prop_df = pd.DataFrame(columns = colnames)
            
        prop_df.loc[cont_id,colnames] = [cont_id] + feat_dict.values()
        cont_df.loc[cont_id] = [cont_id] + [cnt.tolist()]

        cont_id+=1
    print(prop_df.head())
    print(cont_df.head())

    prop_df.to_csv(prop_outpath)
    cont_df.to_csv(cont_outpath)
    
if __name__ == '__main__':
    main()
