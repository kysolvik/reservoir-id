# reservoir-id
Python library for identifying man-made water reservoirs from satellite land cover classification

# Requirements
* NumPy
* pandas
* OpenCV-Python
* GDAL
* scikit-learn
* scikit-image

# Usage
### Required Inputs
1. Landcover classification GeoTIFF. 
2. Intensity greyscale GeoTIFF (e.g. NDVI, NDWI) with same projection, resolution, bounds, etc. as landcover.
3. CSV containing coordinates of positive training examples. Must be same projection as GeoTIFFs.
4. CSV containing coordinates of negative training examples. Must be same projection as GeoTIFFs.

### Morphology
Inputs: landcover tif, flag True/False whether or not to run grow/shrink
Output: GeoTIFF containing water pixels as 1, everything else 0

python grow_shrink_input.py [landcover-tif-path] [run-morphology-True/False] [output-path]

Example:
python grow_shrink_intput.py "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/inputs/xingu_lc.tif" True "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/intermediate/xingu_class_foo.tif" 

### Adding Features
Inputs:
Outputs:

python build_att_table.py "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/intermediate/xingu_class_foo.tif" "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/inputs/ndvi_10m.tif" "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tiles/ndvi_test_v2" True "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/training_points/all_res.csv" "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/training_points/all_nonres.csv" "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tiles/ndvi_test_v2/tables/prop.csv"

### Classification
python classify_reservoirs.py "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tiles/ndvi_test_v2/tables/prop.csv" "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tiles/ndvi_test_v2/tables/classified.csv" 1

### Drawing Outputs
python draw_classified.py "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tiles/ndvi_test_v2/tables/classified.csv" "/Users/ksolvik/Documents/Research/MarciaWork/data/reservoir_id_data/tiles/ndvi_test_v2"
