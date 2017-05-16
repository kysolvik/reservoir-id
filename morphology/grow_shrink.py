#!/usr/bin/env python
"""
@authors: Kylen Solvik, Jordan Graesser
Date Create: 3/8/17

Given land cover tif and water class, outputs tif with only water. 
If morph_yesno is true, will dilate then erode the water objects.

Usage:
python grow_shrink.py [landcover-tif] [morph?(true/fasle)] [output-tif]
"""

import sys
import cv2
import numpy as np
import scipy.misc
import gdal


# Parse command line args
land_cover_path = sys.argv[1]
water_class = 1
morph_truefalse = sys.argv[2]
out_tif = sys.argv[3]

# Function to read in image and save as array
def read_image(lc_path):
    file_handle = gdal.Open(lc_path)
    lc_img = file_handle.GetRasterBand(1).ReadAsArray()
    return(lc_img)

def grow_shrink(lc_array):
    # Recode non-water to 0
    lc_watonly = np.where(lc_array == water_class, 1, 0)
    # Execute shrink/grow
    #se = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # Grow
    lc_watonly = cv2.dilate(np.uint8(lc_watonly), se, iterations=4)
    #Shrink
    lc_watonly = cv2.erode(np.uint8(lc_watonly), se, iterations = 4)
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

    # Grow/shrink
    if morph_truefalse:
        lc = grow_shrink(lc)

    # Print out some basic info
    print(lc.shape)
    print(lc.dtype)
    print(lc.nbytes)
    
    # Write out to geotiff
    write_image(lc,land_cover_path)

    
if __name__ == '__main__':
    main()
