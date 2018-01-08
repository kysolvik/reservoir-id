#!/usr/bin/env python
## SCRIPT ##
"""
@authors: Kylen Solvik
Date Create: 9/25/17
- "Clips" classified raster tif to another raster.
- Retains only regions that contain nonzero value.
- Rasters must have same dimensions and be in same projection.
"""

import gdal
from skimage.measure import label, regionprops
import argparse
from res_modules.res_io import read_write, split_recombine
import numpy as np
import subprocess
import os
import multiprocessing as mp
import glob

#===============================================================================
# Parse command line args
parser = argparse.ArgumentParser(
    description='Clip classified raster using another raster.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('classified_tile_pattern',
                    help='Path to black and white reservoir/non-reservoir tif',
                    type=str)
parser.add_argument('clip_tif',
                    help='Path to intensity image. Usualy NDVI.',
                    type=str)
parser.add_argument('clip_type',
                    help='"pos" or "neg" clip.',
                    type=str)
parser.add_argument('clip_cutoff',
                    help='Keep (pos) or throw out (neg) values below this.',
                    type=int)
parser.add_argument('area_max',
                    help='Maximum area reservoir to keep, in # of pixels.',
                    type=int),
parser.add_argument('output_prefix',
                    help='Prefix for output tifs. If dir, must end with /',
                    type=str)
parser.add_argument('--noclip',
                    help='No clipping, just area filter',
                    action='store_true')
parser.add_argument('--temp_dir',
                    help='Define temporary directory',
                    default=os.path.expanduser('~/temp/'),
                    type=str)
parser.add_argument('--path_prefix',
                    help='To be placed at beginnings of all other path args',
                    default='',
                    type=str)
args = parser.parse_args()

#===============================================================================

def check_delete(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    return()

def reproject_class_tile(tile_path):
    out_tif = args.temp_dir + 'class_temp_' + os.path.basename(tile_path)
    subprocess.call(['gdalwarp','-wm','500','-overwrite','-tr','30','30','-t_srs',
                     '+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 '+
                     '+x_0=0 +y_0=0 +ellps=aust_SA +towgs84=-57,1,-41,0,0,0,0'+
                     '+units=m +no_defs',
                     tile_path,out_tif])
    return(out_tif)

def align_rasters(big_tif,target_tif,class_tile_path):
    out_tif = args.temp_dir + 'lu_temp_' + os.path.basename(class_tile_path)
    data = gdal.Open(target_tif)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    subprocess.call(['gdalwarp','-wm','500','-overwrite','-te',str(minx),str(miny),
                     str(maxx),str(maxy),big_tif,out_tif])
    return(out_tif)

def clip_by_landuse(class_tile_path):
    """
    Clip classified tile by landuse.
    Currently uses global vars from args.
    Could be implemented as a partial function to avoid this.
    """
    # Prep images
    class_tile_reproj = reproject_class_tile(class_tile_path)
    clip_tile = align_rasters(args.clip_tif,class_tile_reproj,
                              class_tile_path)
    
    # Read images
    class_im, geotrans = read_write.read_image(class_tile_reproj)
    clip_im, geotrans = read_write.read_image(clip_tile)
    out_im = np.copy(class_im)
    out_im[out_im!=2] = 0
    out_im[out_im==2] = 1
    
    # Label and get regionprops
    class_labeled = label(out_im)
    plist = regionprops(class_labeled,intensity_image = clip_im)
    
    # Clip outputs
    for i in range(0,len(plist)):
        if not args.noclip:
            if args.clip_type=='pos':
                if plist[i].min_intensity > args.clip_cutoff:
                    out_im[class_labeled == plist[i].label] = 0
            elif args.clip_type=='neg':
                if plist[i].min_intensity <= args.clip_cutoff:
                    out_im[class_labeled == plist[i].label] = 0
        if plist[i].area >= args.area_max:
                out_im[class_labeled == plist[i].label] = 0


    # Write image out
    output_path = args.output_prefix + os.path.basename(class_tile_path)
    read_write.write_image(out_im,class_tile_reproj,output_path,gdal.GDT_Byte)
    check_delete(class_tile_reproj)
    check_delete(clip_tile)
    return()

def main():
    tiles_list = glob.glob(args.classified_tile_pattern)
    print(tiles_list)
    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(clip_by_landuse,tiles_list)
    pool.close()
    pool.join()
    return()

if __name__ == '__main__':
    main()
