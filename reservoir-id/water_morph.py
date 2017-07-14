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
import argparse

from res_modules.res_io import read_write

# Parse command line args
parser = argparse.ArgumentParser(description='Dilate then erode water objects.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('landcover_tif',help='Path to landcover tif with water class',
                    type=str)
parser.add_argument('out_tif',
                    help='Path for output tif with morphed water',
                    type=str)
parser.add_argument('--water_class',
                    help='Integer water class in landcover_tif. Default=1.',
                    default=1,type=int)
parser.add_argument('--no_morph',
                    action="store_true",
                    help='Skip morphology, output unmodifed water objects.')
parser.add_argument('--path_prefix',
                    help='To be placed at beginnings of all other path args',
                    default='',
                    type=str)
args = parser.parse_args()

def grow_shrink(lc_array):
    # Recode non-water to 0
    lc_watonly = np.where(lc_array == args.water_class, 1, 0)
    # Execute shrink/grow
    #se = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # Grow
    lc_watonly = cv2.dilate(np.uint8(lc_watonly), se, iterations=3)
    #Shrink
    lc_watonly = cv2.erode(np.uint8(lc_watonly), se, iterations = 3)
    return(np.uint8(lc_watonly))

def main():
    # Read
    lc,geotrans = read_write.read_image(args.path_prefix + args.landcover_tif)
    print(lc)
    # Grow/shrink
    if not args.no_morph:
        lc = grow_shrink(lc)

    # Print out some basic info
    print(lc.shape)
    print(lc.dtype)
    print(lc.nbytes)
    
    # Write out to geotiff
    read_write.write_image(lc,args.path_prefix + args.landcover_tif,
                           args.path_prefix + args.out_tif,gdal.GDT_Byte)

    
if __name__ == '__main__':
    main()
