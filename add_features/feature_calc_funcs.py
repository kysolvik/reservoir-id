#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 3/8/17
Contains functions needed for building attribute table
"""
import sys
import cv2
import numpy as np
import scipy.misc
import gdal
import pandas as pd
import math

#===============================================================================

# Read in tif as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    gt = file_handle.GetGeoTransform()
    return(file_handle.GetRasterBand(1).ReadAsArray(),gt)

# Function to calculate contour features
def cont_features(obj,shapenames,acut,match_shape):
    # Needed for getting approximate poly
    epsilon = .05*cv2.arcLength(obj,True)
    
    # Basic shape dictionary
    shp_dict = {'obj':obj,
                'approx':cv2.approxPolyDP(obj,epsilon,True),
                'hull':cv2.convexHull(obj),
                'rect':cv2.boxPoints(cv2.minAreaRect(obj))}

    # Object moments
    shp_feats = cv2.moments(obj)

    if shp_feats['m00']>acut:
        shp_feats['obj_area'] = shp_feats['m00']
        return(shp_feats)
    # Match with triangle shape
    for match_meth in range(1,4):
        shp_feats['tri_match'+str(match_meth)] = (
            cv2.matchShapes(obj,match_shape,match_meth,0))


        
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
def calc_nd_feats(obj,nd_image):
    temp_mask = np.zeros(nd_image.shape,np.uint8)
    cv2.drawContours(temp_mask,[obj],0,(255),-1)
    nd_min,nd_max,foo1,foo2 = cv2.minMaxLoc(nd_image,mask = temp_mask)
    nd_mean = int(cv2.mean(nd_image,mask = temp_mask)[0])
    return(nd_min,nd_max,nd_mean)
