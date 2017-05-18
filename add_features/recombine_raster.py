#! /usr/bin/env python

import sys
import os
import gdal

# Args
in_dir = sys.argv[1]
in_prefix = sys.argv[2]
out_tif = sys.argv[3]

# Run command
com_string = "gdalwarp -r near " + in_dir + "/" + in_prefix + "*.tif" \
             + out_tif
os.system(com_string)

print "Done with combine!"
