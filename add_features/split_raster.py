#! /usr/bin/env python

import sys
import os
import gdal

# Args
in_tif = sys.argv[1]
out_dir = sys.argv[2]
out_prefix = sys.argv[3]
tile_size_x = int(sys.argv[4])
tile_size_y = int(sys.argv[5])
overlap_size = int(sys.argv[6])

# Make directory
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
                                 str(i)+ ", " + str(j) + ", " + \
                                 str(tile_size_x) + ", " + str(tile_size_y) \
                                 + " " + str(in_tif) \
                                 + " " + str(out_dir) + str(out_prefix) + \
                                 str(i) + "_" + str(j) + ".tif"
                    os.system(com_string)

print "Done with splitting!"
