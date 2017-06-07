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

def resonly(tile_path, res_class, overwrite):
    if "resonly" not in tile_path:
        tile_noext = path.splitext(tile_path)[0]
        if overwrite or not path.isfile(tile_noext + "_resonly.tif"):
            os.system("gdal_calc.py -A " + tile_path + " --calc='1*(A==" +
                      res_class + ")' --overwrite " +
                      "--creation-option='COMPRESS=LZW' " +
                      "--outfile=" + tile_noext + "_resonly.tif")
    return()

def combine_classified(class_pattern,res_class,output_vrt,overwrite_resonly):
    partial_resonly = partial(resonly, res_class = res_class, overwrite = overwrite_resonly)
    pool = mp.Pool(mp.cpu_count()-2)
    pool.map(partial_resonly,glob.glob(class_pattern))
    pool.close()
    pool.join()

    # Build vrt
    resonly_pattern = path.dirname(class_pattern) + "/*_resonly.tif"
    os.system("gdalbuildvrt " + output_vrt + " " + resonly_pattern)
    return()

def main():
    combine_classified(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]=="True")
    return()

if __name__ == '__main__':
    main()
