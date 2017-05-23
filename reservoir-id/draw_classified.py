#!/usr/bin/env python
## SCRIPT ##
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import gdal
from os import path
import pandas as pd

from res_modules.res_io import read_write, split_recombine

classified_csv_path = sys.argv[1]
tile_dir_path = sys.argv[2]

def draw_classified(classified_csv,tile_dir):

    # Get CSV
    class_df = pd.read_csv(classified_csv,header=0)
    tile_ids_all = np.array([ i.split('-')[0] for i in class_df['id'] ])
    tile_ids_unique = np.unique(tile_ids_all)
    reg_nums = np.array([ int(i.split('-')[1]) for i in class_df['id'] ])
    predictions = np.array((class_df['rf_pred']))
    for tile in tile_ids_unique:
        pos_regions = reg_nums[(tile_ids_all == tile) & (predictions == 2)]
        neg_regions = reg_nums[(tile_ids_all == tile) & (predictions == 1)]

        pos_temp_val = 3
        neg_temp_val = 2

        wat_im_path = tile_dir + "/" + "water/water_" + tile + ".tif"
        wat_im, foo = read_write.read_image(wat_im_path)
        labeled_im_path = tile_dir + "/" + "labeled/labeled_" + tile + ".tif"
        labeled_im, foo = read_write.read_image(labeled_im_path)

        for i in pos_regions:
            wat_im[labeled_im == i] = pos_temp_val

        for i in neg_regions:
            wat_im[labeled_im == i] = neg_temp_val

        # Write out result
        if not os.path.exists(tile_dir+"/classified"):
            os.makedirs(tile_dir+"/classified")

        read_write.write_image(wat_im,wat_im_path,tile_dir + \
                    "/classified/classified_" + tile,gdal.GDT_Byte)

    # Combine them back together
    # split_recombine.recombine_raster(tile_dir + "/classified","classified_",output_tif)

def main():
    draw_classified(classified_csv_path,tile_dir_path)
    return()

if __name__ == '__main__':
    main()
