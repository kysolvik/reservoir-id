#!/usr/bin/env python

import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import gdal
from os import path
import glob
import multiprocessing as mp
from functools import partial
import argparse

# Parse input arguments
parser = argparse.ArgumentParser(description='Creates mosaic of ONLY reservoir objects.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('classified_im_pattern',
                    help='Pattern for glob to find classified images',
                    type=str)
parser.add_argument('res_mosaic_tif',
                    help='Tif for saving mosaiced reservoir image',
                    type=str)
parser.add_argument('--reservoir_class',
                    help='Integer for reservoir class in classified images',
                    type=str,default='3')
parser.add_argument('--path_prefix',
                    help='To be placed at beginnings of all other path args',
                    type=str,default='')
args = parser.parse_args()


def resonly(tile_path, res_class):
    if "resonly" not in tile_path:
        tile_noext = path.splitext(tile_path)[0]
        os.system("gdal_calc.py -A " + tile_path + " --calc='1*(A==" +
                  res_class + ")' --overwrite " +
                  "--creation-option='COMPRESS=LZW' " +
                  "--outfile=" + tile_noext + "_resonly.tif")
    return()

def combine_classified(file_pattern,res_class,output_tif):
    partial_resonly = partial(resonly, res_class = res_class)
    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(partial_resonly,glob.glob(file_pattern))
    pool.close()
    pool.join()

    # Build vrt
    temp_vrt =  path.splitext(output_tif)[0] + '.vrt'
    resonly_pattern = path.dirname(args.classified_im_pattern) + "/*_resonly.tif"
    os.system("gdalbuildvrt " + temp_vrt + " " + resonly_pattern)

    # Translate to tif
    os.system('gdal_translate -co "COMPRESS=LZW" ' + temp_vrt + ' ' + 
              output_tif)
    return()

def main():
    combine_classified(args.path_prefix + args.classified_im_pattern,
                       args.reservoir_class,
                       args.path_prefix + args.res_mosaic_tif)
    return()

if __name__ == '__main__':
    main()
