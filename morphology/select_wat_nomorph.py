#!/usr/bin/env python
"""
@authors: Kylen Solvik, Jordan Graesser
Date Create: 3/8/17
"""

import sys
import cv2
import numpy as np
import scipy.misc
import gdal


# Set some input variables
land_cover_path = '/Users/ksolvik/Documents/Research/MarciaWork/data/shapeAnalysis/classWater1Try4.tif'
water_class = 1
out_tif ='/Users/ksolvik/Documents/Research/MarciaWork/data/build_attribute_table/wat_nomorph.tif'

# Function to read in image and save as array
def read_image(lc_path):
    file_handle = gdal.Open(lc_path)
    lc_img = file_handle.GetRasterBand(1).ReadAsArray()
    return(lc_img)

def select_wat(lc_array,wclass):
    lc_watonly = np.where(lc_array==wclass,1,0)
    return(np.uint8(lc_watonly))

def write_image(lc_array,in_path):
    # Input
    source_ds = gdal.Open(in_path)
    source_band = source_ds.GetRasterBand(1)
    #x_min, x_max, y_min, y_max = source_band.GetExtent()
    
    # Destination
    dst_filename = out_tif
    y_pixels, x_pixels = lc_array.shape  # number of pixels in x
    driver = gdal.GetDriverByName('GTiff')
    outds = driver.Create(dst_filename,x_pixels, y_pixels, 1,gdal.GDT_Byte)
    outds.GetRasterBand(1).WriteArray(lc_array)

    # Add GeoTranform and Projection
    geotrans=source_ds.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=source_ds.GetProjection() #you can get from a exsited tif or import
    outds.SetGeoTransform(geotrans)
    outds.SetProjection(proj)
    outds.FlushCache()
    outds=None

def main():
    # Read
    lc = read_image(land_cover_path)
    print(lc)
    # Grow/shrink
    lc = select_wat(lc,water_class)
    print(lc.shape)
    print(lc.dtype)
    print(lc.nbytes)
    # Write out to geotiff
    write_image(lc,land_cover_path)

    
if __name__ == '__main__':
    main()
