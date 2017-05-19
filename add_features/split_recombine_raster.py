#! /usr/bin/env python

0;95;0cimport sys
import os
import gdal


def split_raster(in_tif,out_dir,out_prefix,tile_size_x,tile_size_y,
                 overlap_size):
        # Make dir
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
        # Open raster
        ds = gdal.Open(in_tif)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize

        # Create tiles
        for i in range(0, xsize, (tile_size_x-overlap_size)):
                for j in range(0, ysize, (tile_size_y-overlap_size)):
                        com_string = "gdal_translate -of GTIFF -srcwin " + \
                                     str(i) + ", " + str(j) + ", " + \
                                     str(tile_size_x) + ", " \
                                     + str(tile_size_y) \
                                     + " " + str(in_tif) \
                                     + " " + str(out_dir) + "/" \
                                     + str(out_prefix) + \
                                     str(i) + "_" + str(j) + ".tif"
                        os.system(com_string)


        return("Done with split")

def recombine_raster(in_dir,in_prefix,out_tif):
        # Run command
        com_string = "gdalwarp -r near -wm 3000 -overwrite " + in_dir + "/" + in_prefix + "* " \
                     + out_tif
        print(com_string)
        os.system(com_string)
        
        return("Done with combine")