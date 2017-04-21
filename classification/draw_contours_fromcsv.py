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
cont_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/contours_testset.csv'
prop_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/att_table_predict_wndwi.csv'
im_outpath = '/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/output_contours_wndwi.tif'

# Set which column holds the prediction in the prop_df
predict_column = 'rf_pred'

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
    wat_im = read_image(im_path)
    wat_im = np.zeros(wat_im.shape,np.uint8)
    # Read old contours csv
    cont_df = pd.read_csv(cont_path)

    # Read props df for prediction
    prop_df = pd.read_csv(prop_path)

    # Join cont_df and prop_df    
    cont_df = pd.merge(cont_df,prop_df,on='id',how='left')
    # Rename prediction column to 'predict'
    cont_df = cont_df.rename(columns={predict_column:'predict'})

    # Set predict for anything that isn't 1 or 2
    cont_df.loc[pd.isnull(cont_df['predict']),'predict'] = 3

    # Pull out reservoirs
    cont_res = cont_df.loc[cont_df['predict']==2]
    cont_nonres = cont_df.loc[cont_df['predict']==1]
    cont_toobig = cont_df.loc[cont_df['predict']==3]

    
    csv_cont_res = map(np.asarray,map(ast.literal_eval,cont_res['contour'].tolist()))
    csv_cont_nonres = map(np.asarray,map(ast.literal_eval,cont_nonres['contour'].tolist()))
    csv_cont_toobig =  map(np.asarray,map(ast.literal_eval,cont_toobig['contour'].tolist()))

    
    # Draw contours
    cv2.drawContours(wat_im,csv_cont_toobig,-1,(3,0,0),-1)
    cv2.drawContours(wat_im,csv_cont_nonres,-1,(1,0,0),-1)
    cv2.drawContours(wat_im,csv_cont_res,-1,(2,0,0),-1)

    # Write out
    write_image(wat_im,im_path,im_outpath)

    print("done!")
if __name__ == '__main__':
    main()
