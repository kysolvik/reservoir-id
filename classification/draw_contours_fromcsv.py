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
cont_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours.csv'
im_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/output_contours.tif'

# Function to read in image and save as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    return(file_handle.GetRasterBand(1).ReadAsArray())

def main():
    # Read image
    wat_im = read_image(im_path)

    # Get contours
    im2, contours, hierarchy = cv2.findContours(wat_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Read old contours csv
    cont_df = pd.read_csv(cont_path)

    csv_conts = cont_df['contour'].tolist()
    csv_conts_lists = map(ast.literal_eval,csv_conts)
    #print(csv_conts_lists[0:2])
    #print(type(ast.literal_eval(csv_conts[0])))
    csv_conts_replace = [w.replace('\n\n',',') for w in csv_conts]
#    print(csv_conts_replace[0])
    csv_conts_new = map(np.asarray,csv_conts_lists)
    
#    print(csv_conts_new[1:5])
   # csv_conts_int = []
   # for i in csv_conts_new:
   #     csv_conts_int += i.astype(int)
    print(csv_conts_new[1:5])
    print(contours[1:5])

    cv2.drawContours(wat_im,csv_conts_new,-1,(2,255,0),-1)
    cv2.imwrite(im_outpath,wat_im)
    print("done!")
if __name__ == '__main__':
    main()
