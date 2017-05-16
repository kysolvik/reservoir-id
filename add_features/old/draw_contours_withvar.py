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
im_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/shapeAnalysis/wat_only_morph.tif'
cont_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours.csv'
prop_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/att_table_wndwi.csv'
im_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours_ndvi.tif'

# Set which column to use for color
color_column = 'ndwi_min'

# Function to read in image and save as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    return(file_handle.GetRasterBand(1).ReadAsArray())

# Function for writing out image with proper projetion 
def write_image(cont_array,inpath,outpath):
    # Input
    source_ds = gdal.Open(inpath)
    source_band = source_ds.GetRasterBand(1)
    #x_min, x_max, y_min, y_max = source_band.GetExtent()
    
    # Destination
    dst_filename = outpath
    y_pixels, x_pixels = cont_array.shape  # number of pixels in x
    driver = gdal.GetDriverByName('GTiff')
    outds = driver.Create(dst_filename,x_pixels, y_pixels, 1,gdal.GDT_Byte)
    outds.GetRasterBand(1).WriteArray(cont_array)
    
    # Add GeoTranform and Projection
    geotrans=source_ds.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=source_ds.GetProjection() #you can get from a exsited tif or import
    outds.SetGeoTransform(geotrans)
    outds.SetProjection(proj)
    outds.FlushCache()
    outds=None
    return() 

def main():
    # Read image
    blank_im = read_image(im_path)
    blank_im = np.zeros(blank_im.shape)
    # Read old contours csv
    cont_df = pd.read_csv(cont_path)

    # Read props df for prediction
    prop_df = pd.read_csv(prop_path)

    # Join cont_df and prop_df    
    cont_df = pd.merge(cont_df,prop_df,on='id',how='left')

    # Loop over all contours and draw them in color based on value of color column
    for i in range(cont_df.shape[0]):
        cont_color = cont_df[color_column][i]
        cv2.drawContours(blank_im,np.asarray(ast.literal_eval(cont_df['contour'][i])),-1,(cont_color,cont_color,cont_color),thickness=10)


    # Write out
    write_image(blank_im,im_path,im_outpath)

    print("done!")

if __name__ == '__main__':
    main()