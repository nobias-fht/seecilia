import os
import pandas as pd
import numpy as np
from bioio import BioImage
from biapy import BiaPy, build_config
import skimage
import joblib
from object_classifier import ObjectClassifier
from scipy.ndimage import distance_transform_edt
import time
from skan import Skeleton, summarize
from skimage.graph import MCP_Geometric
import napari
import matplotlib.pyplot as plt
from functions import *


# def find_images(root_folder, extensions=('.tif', '.tiff', '.czi', '.png', '.jpg', '.jpeg')):

#     """Recursively find all image files under root_folder.
    
#     Args:
#         root_folder: Path to the top-level folder to search.
#         extensions: Tuple of file extensions to include (case-insensitive).
    
#     Returns:
#         Sorted list of full absolute paths to matching image files.
#     """
#     image_paths = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for fname in filenames:
#             if fname.lower().endswith(extensions):
#                 image_paths.append(os.path.join(dirpath, fname))
#     return sorted(image_paths)

# def smallest_axis_index(shape):
#     smallest = min(shape)
#     return shape.index(smallest)

# def open_and_parse_cilia_im(im_path):

#     filename = os.path.basename(im_path)
#     if filename.lower().endswith('.czi'):
#         im = BioImage(im_path)
#         im_data = im.data
#         im_data = np.squeeze(im_data)
#         z_scale, y_scale, x_scale = im.physical_pixel_sizes.Z, im.physical_pixel_sizes.Y, im.physical_pixel_sizes.X  # returns the Z dimension pixel size as found in the metadata

#         return im_data, (z_scale, y_scale, x_scale)

# def extract_cilia_channel(im, cilia_chan):
#     chan_axis = np.argmin(im.shape)

#     cilia_im = np.take(im, cilia_chan, axis=chan_axis)
#     return cilia_im

# def size_filter_masks(masks, min_area, max_area):
#     masks_removed = []
#     props = skimage.measure.regionprops(masks.astype(int))
#     for prop in props:
#         if prop.area > max_area or prop.area < min_area:
#             masks[masks == prop.label] = 0
#             masks_removed.append(prop.label)
#     return masks, masks_removed

# def calculate_slice_circularity_skimage(mask_3d):
#     """
#     Alternative implementation using scikit-image.
#     """
#     max_pixels = 0
#     best_slice_idx = 0
    
#     # Find slice with most pixels
#     for i in range(mask_3d.shape[0]):
#         slice_pixels = np.sum(mask_3d[i])
#         if slice_pixels > max_pixels:
#             max_pixels = slice_pixels
#             best_slice_idx = i
#     if max_pixels == 0:
#         return 0.0, best_slice_idx, 0, []
    
#     # Get the slice with most pixels
#     best_slice = mask_3d[best_slice_idx]
    
#     # Label connected components
#     labeled = skimage.measure.label(best_slice)
#     props = skimage.measure.regionprops(labeled)
    
#     if not props:
#         return 0.0, best_slice_idx, max_pixels, []
    
#     # Get the largest region
#     largest_region = max(props, key=lambda x: x.area)
    
#     area = largest_region.area
#     perimeter = largest_region.perimeter
    
#     if perimeter == 0:
#         return 0.0, best_slice_idx, area, []
    
#     # Calculate circularity
#     circularity = (4 * np.pi * area) / (perimeter ** 2)
    
#     #calculate average circularity over all slices
#     slicewise_circularity = []
#     for i in range(mask_3d.shape[0]):
#         props = skimage.measure.regionprops(mask_3d[i,:,:])
#         if props[0].area > 10:
#             slicewise_circularity.append((4 * np.pi * props[0].area) / (props[0].perimeter ** 2))


#     return circularity, best_slice_idx, area, slicewise_circularity

# def remove_circular_masks(masks, method='simple', max_circularity=0.5):
#     masks_removed = []
#     labs_copy = np.copy(mask_labs)
#     props = skimage.measure.regionprops(mask_labs.astype(int))
#     for i, prop in enumerate(props):
#         if i > 0:
#             cube = extract_cube_padded(mask_labs, prop.bbox,  prop.label, padding=0)
#             circularity, best_slice_idx, area, slicewise_circularity = calculate_slice_circularity_skimage(cube)
#             if method == 'simple':
#                 if circularity > max_circularity:
#                     labs_copy[labs_copy == prop.label] = 0
#                     masks_removed.append(prop.label)
#             if method == '3D':
        
#                 if len(slicewise_circularity) > 0:
#                     if np.amin(slicewise_circularity) > max_circularity: #all slices above threshold
#                         labs_copy[labs_copy == prop.label] = 0
#                         masks_removed.append(prop.label)
#     return labs_copy, masks_removed

# def extract_cube_padded(image, coords,  label=None, padding=1):
#     if len(coords) != 6:
#         raise ValueError("Coordinates must contain exactly 6 values: (z1, y1, x1, z2, y2, x2)")
    
#     # Unpack coordinates
#     z1, y1, x1, z2, y2, x2 = coords
    
#     # Ensure coordinates are integers
#     z1, y1, x1 = int(z1), int(y1), int(x1)
#     z2, y2, x2 = int(z2), int(y2), int(x2)
    
#     # Ensure min/max order (swap if needed)
#     z_min, z_max = min(z1, z2), max(z1, z2)
#     y_min, y_max = min(y1, y2), max(y1, y2)
#     x_min, x_max = min(x1, x2), max(x1, x2)
    
#     # Add padding
#     z_min_padded = z_min - padding
#     y_min_padded = y_min - padding
#     x_min_padded = x_min - padding
#     z_max_padded = z_max + padding
#     y_max_padded = y_max + padding
#     x_max_padded = x_max + padding
    
#     # Clamp coordinates to image bounds
#     z_min_padded = max(0, z_min_padded)
#     y_min_padded = max(0, y_min_padded)
#     x_min_padded = max(0, x_min_padded)
#     z_max_padded = min(image.shape[0], z_max_padded)
#     y_max_padded = min(image.shape[1], y_max_padded)
#     x_max_padded = min(image.shape[2], x_max_padded)
    
#     # Extract the padded cube
#     cube = np.copy(image[z_min_padded:z_max_padded, 
#                  y_min_padded:y_max_padded, 
#                  x_min_padded:x_max_padded])
#     if label is not None:
#         cube[cube != label] = 0
    
#     return cube

# def get_geodesic_poles(binary_tube):
#     """Finds the two extreme endpoints of a 3D binary tube."""
#     # Step 1: Base cost is 1 for the tube, infinity for background
#     cost_volume = np.where(binary_tube, 1.0, np.inf)
#     mcp = MCP_Geometric(cost_volume)
    
#     # Step 2: Find the geometric center (approximate) to start the search
#     dist = distance_transform_edt(binary_tube)
#     center_idx = tuple(np.argwhere(dist == dist.max())[0])
    
#     # Step 3: Find the furthest point from the center -> Pole A
#     # FIX: Catch the returned arrays from find_costs
#     costs_from_center, _ = mcp.find_costs([center_idx])
#     costs_from_center[np.isinf(costs_from_center)] = -1 # Ignore background
#     pole_a = tuple(np.argwhere(costs_from_center == costs_from_center.max())[0])
    
#     # Step 4: Find the furthest point from Pole A -> Pole B
#     mcp = MCP_Geometric(cost_volume) # Reset MCP
#     # FIX: Catch the returned arrays from find_costs
#     costs_from_a, _ = mcp.find_costs([pole_a])
#     costs_from_a[np.isinf(costs_from_a)] = -1
#     pole_b = tuple(np.argwhere(costs_from_a == costs_from_a.max())[0])
    
#     return pole_a, pole_b

# def extract_exact_centerline(binary_tube, power=2.0):
#     """
#     Extracts the endpoint-preserving centerline and exact length.
#     """
#     # 1. Find the exact endpoints on the boundary
#     pole_a, pole_b = get_geodesic_poles(binary_tube)
    
#     # 2. Compute Distance Transform
#     dist = distance_transform_edt(binary_tube)
#     max_dist = dist.max()
    
#     # 3. Create the Cost Volume for the centerline
#     # Equation: C = (D_max - D + 1)^power
#     # Voxels in the center have cost ~1. Voxels at the edge have high cost.
#     with np.errstate(divide='ignore', invalid='ignore'):
#         cost_volume = np.where(
#             binary_tube, 
#             (max_dist - dist + 1.0) ** power, 
#             np.inf
#         )
    
#     # 4. Route the shortest path through the "valley"
#     mcp = MCP_Geometric(cost_volume)
#     costs, traceback = mcp.find_costs([pole_a], [pole_b])
    
#     # Traceback gives the exact sequence of coordinates
#     path_coords = mcp.traceback(pole_b)
#     path_coords = np.array(path_coords)
    
#     # 5. Calculate precise 3D Euclidean length
#     # Summing the distance between consecutive voxels (accounts for diagonals)
#     diffs = np.diff(path_coords, axis=0)
#     distances = np.linalg.norm(diffs, axis=1)
#     exact_length = np.sum(distances)
    
#     return path_coords, exact_length

# def create_centerline_mask(original_tube, path_coords):
#     """
#     Converts the 1D path coordinates back into a 3D binary mask 
#     of the exact same shape as the input tube.
#     """
#     # Initialize an array of False with the same shape as the input
#     centerline_mask = np.zeros_like(original_tube, dtype=bool)
    
#     # Isolate the Z, Y, and X coordinate arrays
#     # Note: MCP_Geometric returns coordinates in (Z, Y, X) order
#     z_coords = path_coords[:, 0]
#     y_coords = path_coords[:, 1]
#     x_coords = path_coords[:, 2]
    
#     # Set the voxels along the path to True
#     centerline_mask[z_coords, y_coords, x_coords] = True
    
#     return centerline_mask

PATH = '/facility/imganfac/neurogenomics/taverna/scordato_cilia_v2/test_data/czi'   
OUTPUT_PATH = '/facility/imganfac/neurogenomics/taverna/scordato_cilia_v2/test_output_2'
os.makedirs(OUTPUT_PATH, exist_ok=True)
MODEL_PATH = '/facility/imganfac/neurogenomics/taverna/scordato_cilia_v2/model_training_v2/cilia_zoom_semantic_new/checkpoints/cilia_zoom_semantic_new_1-checkpoint-best.pth'
BINARY_THRESHOLD = 0.8
MIN_SEG_SIZE = 50
MAX_CILIA_LENGTH = 4
CLASSIFIER_FOLDER = '/facility/imganfac/neurogenomics/taverna/scordato/classifiers'
MIN_SEG_SIZE = 50
WINDOW = 5
POLYORDER = 2
elem = skimage.morphology.footprint_rectangle((3,3,3))
files = os.listdir(PATH)
plt.style.use("/facility/imganfac/Damian/facility-sop/IAF_style.mplstyle")

filenum = 1

###EXTRACT AND PREPROCESS
filename = files[filenum]
file_path = os.path.join(PATH, files[filenum])
print(filename)
im, scale = open_and_parse_cilia_im(file_path)

cilia_im = extract_cilia_channel(im, 0)
print('cilia_im extracted, beginning inference step')

#skimage.io.imsave(os.path.join(OUTPUT_PATH, 'cilia_im.tif'), cilia_im, check_contrast=False)


###BIAPY SEGMENT


biapy = BiaPy.load_workflow_from_model(MODEL_PATH, name="infer", gpu="0")

print('model_loaded, beginning inference')
mask = biapy.predict(cilia_im, verbose=False)
print(mask.shape)
# for file in files:
print('inference complete')
mask = np.squeeze(mask)

skimage.io.imsave(os.path.join(OUTPUT_PATH, 'cilia_mask.tif'), mask, check_contrast=False)
skimage.io.imsave(os.path.join(OUTPUT_PATH, 'cilia_raw.tif'), cilia_im, check_contrast=False)

#mask = skimage.io.imread(os.path.join(OUTPUT_PATH, 'cilia_mask.tif'))
#print(mask.shape)
#skimage.io.imsave(os.path.join(OUTPUT_PATH, 'cilia_mask.tif'), mask, check_contrast=False)

masks_thresholded = mask >= BINARY_THRESHOLD
masks_median = skimage.filters.median(masks_thresholded, footprint=elem)
masks_filled = skimage.morphology.remove_small_holes(masks_median > 0, max_size=16)
mask_labs = skimage.measure.label(masks_filled)
skimage.io.imsave(os.path.join(OUTPUT_PATH, 'cilia_mask_labeled.tif'), mask_labs.astype(np.uint16), check_contrast=False)
labs_large_removed, masks_removed = size_filter_masks(mask_labs, 100, 5000)
skimage.io.imsave(os.path.join(OUTPUT_PATH, 'cilia_mask_labeled_large_removed.tif'), labs_large_removed.astype(np.uint16), check_contrast=False)

labs_clean, masks_removed = remove_circular_masks(labs_large_removed, method='3D', max_circularity=0.95)
skimage.io.imsave(os.path.join(OUTPUT_PATH, 'cilia_mask_labeled_circ_removed.tif'), labs_clean.astype(np.uint16), check_contrast=False)

#CLASSIFIER
if 'zoom' in filename:
    class_name = 'mouse_zoom.joblib'
    print('loading classifier: ' + class_name)
    clf = joblib.load(os.path.join(CLASSIFIER_FOLDER, class_name))
else:
    class_name = 'mouse_nozoom.joblib'
    print('loading classifier: ' + class_name)
    clf = joblib.load(os.path.join(CLASSIFIER_FOLDER, class_name))

result, class_names = clf.predict(cilia_im, labs_clean)
skimage.io.imsave(os.path.join(OUTPUT_PATH, 'class_raw.tif'), result.astype(np.uint16), check_contrast=False)

result[result == 1] = 0

simple_masks_thresholded = labs_clean * result

skimage.io.imsave(os.path.join(OUTPUT_PATH, 'class.tif'), result, check_contrast=False)
skimage.io.imsave(os.path.join(OUTPUT_PATH, 'simple_masks_classified.tif'), simple_masks_thresholded, check_contrast=False)

#SKELETONIZATION

props = skimage.measure.regionprops(simple_masks_thresholded.astype(int))  
raw_skeletons = np.zeros(simple_masks_thresholded.shape, dtype=np.uint16)
simple_skel = np.zeros(simple_masks_thresholded.shape, dtype=np.uint16)

for prop in props:
    label = prop.label
    if prop.label == 14:
        print(prop.area)
    cube_raw = extract_cube_padded(cilia_im, prop.bbox, prop.label, padding=0)
    cube = extract_cube_padded(simple_masks_thresholded, prop.bbox, prop.label, padding=0)
    if np.sum(cube>0) < MIN_SEG_SIZE:
        continue
    skel_skeleton = skimage.morphology.skeletonize(cube>0)
    if np.count_nonzero(skel_skeleton) < 2:
        continue
    centerline_coords, length = extract_centerline_anisotropic(cube, power=2.0, spacing=scale)
    center_mask = create_centerline_mask(cube, centerline_coords)
    if np.count_nonzero(center_mask) < 3:
        continue
        
    raw_skeletons[prop.bbox[0]:prop.bbox[3], prop.bbox[1]:prop.bbox[4], prop.bbox[2]:prop.bbox[5]] += center_mask.astype(np.uint16) * prop.label
    branch_data = summarize(Skeleton(center_mask), separator='-')
    if len(branch_data) == 1:
        simple_skel[prop.bbox[0]:prop.bbox[3], prop.bbox[1]:prop.bbox[4], prop.bbox[2]:prop.bbox[5]] += center_mask.astype(np.uint16) * prop.label

props = skimage.measure.regionprops(simple_skel.astype(int))
labels = []
branch_lengths_euc = []
branch_lengths_path = []
spline_lengths = []
sampled_coords = []
sampled_coords_pixel_list = []


pixel_diag = np.linalg.norm(scale)

for prop in props:
            

    #extract cubes
    cube_skel = extract_cube_padded(simple_skel, prop.bbox, label=prop.label, padding=0)
    

    skeleton = Skeleton(cube_skel,spacing=scale) 
    branch_data = summarize(Skeleton(cube_skel, spacing=scale), separator='-')
    if len(branch_data) > 1:
        print(f"Warning: More than one branch found for label {prop.label} in file {filename} prop {prop.label}. Ignoring.")
        continue
    crop_coords = skeleton.path_coordinates(0)
    coords = [
        [z + int(prop.bbox[0]), y + int(prop.bbox[1]), x + int(prop.bbox[2])]
        for z, y, x in crop_coords
    ]

    coords = coords * np.array(scale)
 
    result = calc_skeleton_curve(
        coords,
        sampling_dist=pixel_diag / 16,
        savgol_window=WINDOW,
        savgol_polyorder=POLYORDER,
        )   

    sampled_coords_pixel = result["sampled_coords"] / np.array(scale)
    labels.append(prop.label)
    spline_lengths.append(result['length'])
    branch_lengths_euc.append(branch_data['euclidean-distance'].values[0])
    branch_lengths_path.append(branch_data['branch-distance'].values[0])
    sampled_coords.append(result['sampled_coords'])
    sampled_coords_pixel_list.append(sampled_coords_pixel)
df = pd.DataFrame(columns=['roi_label', 'spline_length', 'dist_euc', 'dist_path', 'sampled_coords'])
df['roi_label'] = labels
df['spline_length'] = spline_lengths
df['dist_euc'] = branch_lengths_euc
df['dist_path'] = branch_lengths_path
df['sampled_coords'] = sampled_coords
df['sampled_coords_pixel'] = sampled_coords_pixel_list
df.to_csv(os.path.join(OUTPUT_PATH, 'quantification.csv'), index=False)

make_plot(df)

all_sampled_points = np.vstack(sampled_coords_pixel_list) if sampled_coords_pixel_list else np.empty((0, 3))

viewer = napari.Viewer()
viewer.add_image(cilia_im)
viewer.add_labels(simple_masks_thresholded, name='simple_masks_thresholded')
viewer.add_labels(simple_skel, name='simple_skeletons')
viewer.add_points(all_sampled_points, name='sampled_coords_pixel', size=2, face_color='cyan')
napari.run()