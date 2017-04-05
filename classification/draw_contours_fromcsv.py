#!/usr/bin/env python
"""
@authors: Kylen Solvik
Date Create: 4/4/17
"""

import sys
import cv2
import numpy as np
import scipy.misc
import gdal
import pandas as pd
import ast

# Set some input variables
im_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/wat_only_clip.tif'
cont_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours_foodist.csv'
prop_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/att_table_foodist.csv'
im_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/output_contours.tif'

# Function to read in image and save as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    return(file_handle.GetRasterBand(1).ReadAsArray())

def main():
    # Read image
    wat_im = read_image(im_path)


    # Read old contours csv
    cont_df = pd.read_csv(cont_path)

    # Read props df for prediction
    prop_df = pd.read_csv(prop_path)

    cont_df['predict'] = prop_df['class']

    cont_res = cont_df.loc[cont_df['predict']==2]
    cont_nonres = cont_df.loc[cont_df['predict']==1]
    cont_other = cont_df.loc[cont_df['predict']<1]
    
    csv_cont_res = map(np.asarray,map(ast.literal_eval,cont_res['contour'].tolist()))
    csv_cont_nonres = map(np.asarray,map(ast.literal_eval,cont_nonres['contour'].tolist()))
    csv_cont_other =  map(np.asarray,map(ast.literal_eval,cont_other['contour'].tolist()))
    
    # Draw contours
    cv2.drawContours(wat_im,csv_cont_other,-1,(1,255,0),-1)
    cv2.drawContours(wat_im,csv_cont_nonres,-1,(2,255,0),-1)
    cv2.drawContours(wat_im,csv_cont_res,-1,(3,255,0),-1)

    # Write out
    cv2.imwrite(im_outpath,wat_im)
    print("done!")
if __name__ == '__main__':
    main()
