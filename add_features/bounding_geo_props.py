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
training_csv_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/training_points/'
prop_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/att_table.csv'
cont_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours.csv'
shapenames = ['obj','approx','hull','rect']

triangle_shape = np.asarray([[[0,0]], [[4,0]], [[2,12]]])

area_cutoff = 500000
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
    shp_dict = {'obj':obj,'approx':cv2.approxPolyDP(obj,epsilon,True),'hull':cv2.convexHull(obj),'rect':cv2.boxPoints(cv2.minAreaRect(obj))}

    # Object moments
    shp_feats = cv2.moments(obj)

    if shp_feats['m00']>area_cutoff:
        shp_feats['obj_area'] = shp_feats['m00']
        return(shp_feats)
    # Match with triangle shape
    for match_meth in range(1,4):
        shp_feats['tri_match'+str(match_meth)] = cv2.matchShapes(obj,triangle_shape,match_meth,0)


        
    # Get areas and all that good stuff
    for shp in shapenames :
        shp_feats[shp + '_area'] = cv2.contourArea(shp_dict[shp])
        shp_feats[shp + '_perim'] = cv2.arcLength(shp_dict[shp],True)
        
    # Circle is separate because it's a little messier
    (circx,circy),circradius = cv2.minEnclosingCircle(obj)
    shp_feats['circ_rad'] = circradius

    # Return dictionary
    return(shp_feats)
    
# Funciton to caclulate derived features
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
    

def main():
    # Read image
    wat_im,geotrans = read_image(im_path)

    # Read in training points csv
    res_csv = np.genfromtxt(training_csv_path + "all_res.csv",delimiter=",",skip_header=1)
    nonres_csv = np.genfromtxt(training_csv_path + "all_nonres.csv",delimiter=",",skip_header=1)
    
    # Get contours
    im2, contours, hierarchy = cv2.findContours(wat_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print('Number of Objects:' + str(len(contours)))
    # Create array for storing contours
    cont_df = pd.DataFrame(columns=['id','contour'])
    cont_df['contour'].astype(object)
    cont_id = 0
    print_id = 0
    for cnt in contours:
        feat_dict = cont_features(cnt)
        if feat_dict['obj_area']==0:
            continue
        elif feat_dict['obj_area']>area_cutoff:
            prop_df.loc[cont_id,['id','obj_area']] = [cont_id] + [feat_dict['obj_area']]
            print("Too Big!")
        else:
            feat_dict = derived_features(feat_dict)

            # Get class
            feat_class = set_train_class(cnt,res_csv,nonres_csv,geotrans) 
            
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
