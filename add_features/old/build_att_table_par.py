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
from joblib import Parallel, delayed
import multiprocessing
import time

#===============================================================================

# Set some input variables
lead_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/'
im_path = lead_path + 'build_attribute_table/wat_only_morph.tif'
ndwi_path = lead_path + 'build_attribute_table/ndwi_10m.tif'
training_csv_path = lead_path + 'build_attribute_table/training_points/'

# Output paths
prop_outpath = lead_path + 'build_attribute_table/att_table_wndwi.csv'
cont_outpath = lead_path + 'build_attribute_table/contours.csv'

# Some parameters for the calculating shape descriptors
triangle_shape = np.asarray([[[0,0]], [[4,0]], [[2,12]]])
area_cutoff = 500000
shapenames = ['obj','approx','hull','rect']

#===============================================================================

# Function to read in image and save as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    gt = file_handle.GetGeoTransform()
    #print(gt)
    return(file_handle.GetRasterBand(1).ReadAsArray(),gt)

# Function to calculate contour features
def cont_features(obj):
    # Needed for getting approximate poly
    epsilon = .05*cv2.arcLength(obj,True)
    
    # Basic shape dictionary
    shp_dict = {'obj':obj,
                'approx':cv2.approxPolyDP(obj,epsilon,True),
                'hull':cv2.convexHull(obj),
                'rect':cv2.boxPoints(cv2.minAreaRect(obj))}

    # Object moments
    shp_feats = cv2.moments(obj)

    if shp_feats['m00']>area_cutoff:
        shp_feats['obj_area'] = shp_feats['m00']
        return(shp_feats)
    # Match with triangle shape
    for match_meth in range(1,4):
        shp_feats['tri_match'+str(match_meth)] = (
            cv2.matchShapes(obj,triangle_shape,match_meth,0))


        
    # Get areas and all that good stuff
    for shp in shapenames :
        shp_feats[shp + '_area'] = cv2.contourArea(shp_dict[shp])
        shp_feats[shp + '_perim'] = cv2.arcLength(shp_dict[shp],True)
        
    # Circle is separate because it's a little messier
    (circx,circy),circradius = cv2.minEnclosingCircle(obj)
    shp_feats['circ_rad'] = circradius

    # Return dictionary
    return(shp_feats)
    
# Function to caclulate derived features
def derived_features(fd):
    fd['circ_rsq'] = math.pow(fd['circ_rad'],2)
    fd['eq_diam'] = math.sqrt((4*fd['obj_area'])/math.pi)
    fd['solidity'] = fd['obj_area']/fd['hull_area']
    return(fd)

# Function for converting csv points (in same projection as image) to x,y pixels
def get_pixel_xy(gt,pos):
    x=int((pos[0] - gt[0])/gt[1])
    y=int((pos[1] - gt[3])/gt[5])
    #print x,y
    return(x,y)
            
# Function to check if contour contains either a res or nonres point
def set_train_class(obj,res_p,nonres_p,gt):
    obj_class = 0
    for row in res_p:
        x_i,y_i = get_pixel_xy(gt,row[0:2])
        point_poly_test = cv2.pointPolygonTest(obj,(x_i,y_i),False)
        if point_poly_test >= 0:
            obj_class = 2
            break
        
    for row in nonres_p:
        x_i,y_i = get_pixel_xy(gt,row[0:2])
        point_poly_test = cv2.pointPolygonTest(obj,(x_i,y_i),False)
        if point_poly_test >= 0:
            obj_class = 1
            break
    return(obj_class)

# Function to calculate NDWI derived features
def calc_nd_feats(obj,nd_image,mask):
    cv2.drawContours(mask,obj,-1,(255),1)
    nd_min,nd_max,foo1,foo2 = cv2.minMaxLoc(nd_image,mask = mask)
    nd_mean = int(cv2.mean(nd_image,mask = mask)[0])
    return(nd_min,nd_max,nd_mean)

def process_contour(index,conts):
    cnt = conts[index]
    feat_dict = cont_features(cnt)
#    if feat_dict['obj_area']==0:
 #       return()
 #   elif feat_dict['obj_area']>area_cutoff:
 #       print("Too Big!")
 #       return()
        
  #  else:
    #    # Calculate extra features
    #    feat_dict = derived_features(feat_dict)
    return [feat_dict.values()]

#===============================================================================

def main():
    # Read images
    wat_im,geotrans = read_image(im_path)
    ndwi_im, ndwi_geotrans = read_image(ndwi_path)
    mask_im  = np.zeros(wat_im.shape,np.uint8)
    # Read in training points csv
    res_csv = np.genfromtxt(training_csv_path + "all_res.csv",
                            delimiter=",",skip_header=1)
    nonres_csv = np.genfromtxt(training_csv_path + "all_nonres.csv",
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
    t1 = time.time()
    results = Parallel(n_jobs=4) (delayed(process_contour) (i,contours) for i in range(10))
    print 'parallel', time.time() - t1

    t2 = time.time()
    for i in range(10):
        process_contour(i,contours)
    print 'nonpar', time.time() - t2
    print(results)
#    prop_df.to_csv(prop_outpath,index=False)
#    cont_df.to_csv(cont_outpath,index=False)
    
if __name__ == '__main__':
    main()
