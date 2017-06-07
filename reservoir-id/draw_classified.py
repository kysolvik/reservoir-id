#!/usr/bin/env python

import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import gdal
from os import path
import pandas as pd
import multiprocessing as mp
from functools import partial

from res_modules.res_io import read_write, split_recombine

classified_csv_path = sys.argv[1]
tile_dir_path = sys.argv[2]
predict_column = 'clf_pred'

def draw_single_tile(tile,tile_ids_all,reg_nums,predictions,tile_dir):
    pos_regions = reg_nums[(tile_ids_all == tile) & (predictions == 2)]
    neg_regions = reg_nums[(tile_ids_all == tile) & (predictions == 1)]

    pos_temp_val = 3
    neg_temp_val = 2
    #wat_im_path = tile_dir + "/" + "water/water_" + tile + ".tif"
    #wat_im, foo = read_write.read_image(wat_im_path)
    labeled_im_path = tile_dir + "/" + "labeled/labeled_" + tile + ".tif"
    labeled_im, foo = read_write.read_image(labeled_im_path)
    wat_im = np.copy(labeled_im)
    wat_im[np.nonzero(wat_im)] = 1
    
    
    for i in pos_regions:
        wat_im[labeled_im == i] = pos_temp_val
        
    for i in neg_regions:
        wat_im[labeled_im == i] = neg_temp_val

    # Write out result
    if not os.path.exists(tile_dir+"/classified"):
        os.makedirs(tile_dir+"/classified")
    
    read_write.write_image(wat_im,labeled_im_path,tile_dir + \
                           "/classified/classified_" + tile + ".tif",gdal.GDT_Byte)
              
def draw_classified(classified_csv,tile_dir):

    # Get CSV
    class_df = pd.read_csv(classified_csv,header=0)
    tile_ids_all = np.array([ i.split('-')[0] for i in class_df['id'] ])
    tile_ids_unique = np.unique(tile_ids_all)
    reg_nums = np.array([ int(i.split('-')[1]) for i in class_df['id'] ])
    predictions = np.array((class_df[predict_column]))
    partial_draw = partial(draw_single_tile,tile_ids_all=tile_ids_all,
                           reg_nums=reg_nums,predictions=predictions,
                           tile_dir=tile_dir)
    pool = mp.Pool(mp.cpu_count()-2)
    pool.map(partial_draw,tile_ids_unique)
    pool.close()
    pool.join()
    # Combine them back together
    # split_recombine.recombine_raster(tile_dir + "/classified","classified_",output_tif)

def main():
    draw_classified(classified_csv_path,tile_dir_path)
    return()

if __name__ == '__main__':
    main()
