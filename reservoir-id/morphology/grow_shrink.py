#!/usr/bin/env python
## SCRIPT ##

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

from ..io import read_write

# Parse command line args
land_cover_path = sys.argv[1]
water_class = 1
morph_truefalse = sys.argv[2]
out_tif = sys.argv[3]

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

def main():
    # Read
    lc = read_write.read_image(land_cover_path)

    # Grow/shrink
    if morph_truefalse:
        lc = grow_shrink(lc)

    # Print out some basic info
    print(lc.shape)
    print(lc.dtype)
    print(lc.nbytes)
    
    # Write out to geotiff
    read_write.write_image(lc,land_cover_path)

    
if __name__ == '__main__':
    main()
