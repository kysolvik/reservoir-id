#!/usr/bin/env python

import sys
import os
import math
import numpy as np
import gdal
from os import path
import pandas as pd

# Read in tif as array
def read_image(filepath):
    file_handle = gdal.Open(filepath)
    gt = file_handle.GetGeoTransform()
    return(file_handle.GetRasterBand(1).ReadAsArray(),gt)

# Function for writing out image with proper projetion
def write_image(im,inpath,outpath,gdal_dtype):
    
    # Input
    source_ds = gdal.Open(inpath)
    source_band = source_ds.GetRasterBand(1)
    #x_min, x_max, y_min, y_max = source_band.GetExtent()
    
    # Destination
    dst_filename = outpath
    y_pixels, x_pixels = im.shape  # number of pixels in x
    driver = gdal.GetDriverByName('GTiff')
    outds = driver.Create(dst_filename,x_pixels, y_pixels, 1,gdal_dtype)
    outds.GetRasterBand(1).WriteArray(im)
    
    # Add GeoTranform and Projection
    geotrans=source_ds.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=source_ds.GetProjection() #you can get from a exsited tif or import
    outds.SetGeoTransform(geotrans)
    outds.SetProjection(proj)
    outds.FlushCache()
    outds=None
    return()
