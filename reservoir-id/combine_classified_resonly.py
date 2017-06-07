#!/usr/bin/env python

import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import gdal
from os import path
import glob

def combine_classified(class_pattern,res_class,output_vrt):
    # First, gdal calc all outputs to only include reservoirs
    for tile_path in glob.glob(class_pattern):
        os.system("gdal_calc.py -A " + tile_path + " --calc='1*(A==" +
                  res_class + ")' --overwrite " +
                  "--creation-option='COMPRESS=LZW' " +
                  "--outfile=" + tile_path)
    # Build vrt
    os.system("gdalbuildvrt " + output_vrt + " " + class_pattern)
              
def main():
    combine_classified(sys.argv[1],sys.argv[2],sys.argv[3])
    return()

if __name__ == '__main__':
    main()
