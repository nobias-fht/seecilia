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
from object_classifier import ObjectClassifier
from scipy.signal import savgol_filter
from bezier import PiecewiseBezier
from typing import TypedDict
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def find_images(root_folder, extensions=('.tif', '.tiff', '.czi', '.png', '.jpg', '.jpeg')):

    """Recursively find all image files under root_folder.
    
    Args:
        root_folder: Path to the top-level folder to search.
        extensions: Tuple of file extensions to include (case-insensitive).
    
    Returns:
        Sorted list of full absolute paths to matching image files.
    """
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, fname))
    return sorted(image_paths)

def smallest_axis_index(shape):
    smallest = min(shape)
    return shape.index(smallest)

def open_and_parse_cilia_im(im_path):

    filename = os.path.basename(im_path)
    if filename.lower().endswith('.czi'):
        im = BioImage(im_path)
        im_data = im.data
        im_data = np.squeeze(im_data)
        z_scale, y_scale, x_scale = im.physical_pixel_sizes.Z, im.physical_pixel_sizes.Y, im.physical_pixel_sizes.X  # returns the Z dimension pixel size as found in the metadata

        return im_data, (z_scale, y_scale, x_scale)

    if filename.lower().endswith('lif'):
        im = BioImage(im_path)
        im_data = im.data
        im_data = np.squeeze(im_data)
        z_scale, y_scale, x_scale = im.physical_pixel_sizes.Z, im.physical_pixel_sizes.Y, im.physical_pixel_sizes.X  # returns the Z dimension pixel size as found in the metadata

        return im_data, (z_scale, y_scale, x_scale)

def extract_cilia_channel(im, cilia_chan):
    chan_axis = np.argmin(im.shape)

    cilia_im = np.take(im, cilia_chan, axis=chan_axis)
    return cilia_im

def size_filter_masks(masks, min_area, max_area):
    masks_removed = []
    props = skimage.measure.regionprops(masks.astype(int))
    for prop in props:
        if prop.area > max_area or prop.area < min_area:
            masks[masks == prop.label] = 0
            masks_removed.append(prop.label)
    return masks, masks_removed

def calculate_slice_circularity_skimage(mask_3d):
    """
    Alternative implementation using scikit-image.
    """
    max_pixels = 0
    best_slice_idx = 0
    
    # Find slice with most pixels
    for i in range(mask_3d.shape[0]):
        slice_pixels = np.sum(mask_3d[i])
        if slice_pixels > max_pixels:
            max_pixels = slice_pixels
            best_slice_idx = i
    if max_pixels == 0:
        return 0.0, best_slice_idx, 0, []
    
    # Get the slice with most pixels
    best_slice = mask_3d[best_slice_idx]
    
    # Label connected components
    labeled = skimage.measure.label(best_slice)
    props = skimage.measure.regionprops(labeled)
    
    if not props:
        return 0.0, best_slice_idx, max_pixels, []
    
    # Get the largest region
    largest_region = max(props, key=lambda x: x.area)
    
    area = largest_region.area
    perimeter = largest_region.perimeter
    
    if perimeter == 0:
        return 0.0, best_slice_idx, area, []
    
    # Calculate circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    #calculate average circularity over all slices
    slicewise_circularity = []
    for i in range(mask_3d.shape[0]):
        props = skimage.measure.regionprops(mask_3d[i,:,:])
        if props[0].area > 10:
            slicewise_circularity.append((4 * np.pi * props[0].area) / (props[0].perimeter ** 2))


    return circularity, best_slice_idx, area, slicewise_circularity

def remove_circular_masks(masks, method='simple', max_circularity=0.5, max_axis = 0.8):
    masks_removed = []
    labs_copy = np.copy(masks)
    props = skimage.measure.regionprops(masks.astype(int))
    for i, prop in enumerate(props):
        if i > 0:
            cube = extract_cube_padded(masks, prop.bbox,  prop.label, padding=0)
            
            if method != 'axes':
                circularity, best_slice_idx, area, slicewise_circularity = calculate_slice_circularity_skimage(cube)
                if method == 'simple':
                    if circularity > max_circularity:
                        labs_copy[labs_copy == prop.label] = 0
                        masks_removed.append(prop.label)
                if method == '3D':
            
                    if len(slicewise_circularity) > 0:
                        if np.amin(slicewise_circularity) > max_circularity: #all slices above threshold
                            labs_copy[labs_copy == prop.label] = 0
                            masks_removed.append(prop.label)
            if method == 'axes':
                # Implement axes-based circularity check if needed
                proj = np.max(cube, axis=0)  # 2D projection along Z-axis
                props = skimage.measure.regionprops(skimage.measure.label(proj))
                major_axis = props[0].axis_major_length if props else 0
                minor_axis = props[0].axis_minor_length if props else 0
                if major_axis > 0 and minor_axis > 0:
                    axis_circularity = minor_axis / major_axis
                    if axis_circularity > max_axis:
                        labs_copy[labs_copy == prop.label] = 0
                        masks_removed.append(prop.label)
                pass
    return labs_copy, masks_removed

def remove_circular_masks_old(masks, method='simple', max_circularity=0.5):
    masks_removed = []
    labs_copy = np.copy(masks)
    props = skimage.measure.regionprops(masks)
    for i, prop in enumerate(props):
        if i > 0:
            cube = extract_cube_padded(masks, prop.bbox,  prop.label, padding=0)
            circularity, best_slice_idx, area, slicewise_circularity = calculate_slice_circularity_skimage(cube)
            if method == 'simple':
                if circularity > max_circularity:
                    labs_copy[labs_copy == prop.label] = 0
                    masks_removed.append(prop.label)
            if method == '3D':
        
                if len(slicewise_circularity) > 0:
                    if np.amin(slicewise_circularity) > max_circularity: #all slices above threshold
                        labs_copy[labs_copy == prop.label] = 0
                        masks_removed.append(prop.label)
    return labs_copy, masks_removed

def extract_cube_padded(image, coords,  label=None, padding=1):
    if len(coords) != 6:
        raise ValueError("Coordinates must contain exactly 6 values: (z1, y1, x1, z2, y2, x2)")
    
    # Unpack coordinates
    z1, y1, x1, z2, y2, x2 = coords
    
    # Ensure coordinates are integers
    z1, y1, x1 = int(z1), int(y1), int(x1)
    z2, y2, x2 = int(z2), int(y2), int(x2)
    
    # Ensure min/max order (swap if needed)
    z_min, z_max = min(z1, z2), max(z1, z2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    x_min, x_max = min(x1, x2), max(x1, x2)
    
    # Add padding
    z_min_padded = z_min - padding
    y_min_padded = y_min - padding
    x_min_padded = x_min - padding
    z_max_padded = z_max + padding
    y_max_padded = y_max + padding
    x_max_padded = x_max + padding
    
    # Clamp coordinates to image bounds
    z_min_padded = max(0, z_min_padded)
    y_min_padded = max(0, y_min_padded)
    x_min_padded = max(0, x_min_padded)
    z_max_padded = min(image.shape[0], z_max_padded)
    y_max_padded = min(image.shape[1], y_max_padded)
    x_max_padded = min(image.shape[2], x_max_padded)
    
    # Extract the padded cube
    cube = np.copy(image[z_min_padded:z_max_padded, 
                 y_min_padded:y_max_padded, 
                 x_min_padded:x_max_padded])
    if label is not None:
        cube[cube != label] = 0
    
    return cube

def get_geodesic_poles(binary_tube):
    """Finds the two extreme endpoints of a 3D binary tube."""
    # Step 1: Base cost is 1 for the tube, infinity for background
    cost_volume = np.where(binary_tube, 1.0, np.inf)
    mcp = MCP_Geometric(cost_volume)
    
    # Step 2: Find the geometric center (approximate) to start the search
    dist = distance_transform_edt(binary_tube)
    center_idx = tuple(np.argwhere(dist == dist.max())[0])
    
    # Step 3: Find the furthest point from the center -> Pole A
    # FIX: Catch the returned arrays from find_costs
    costs_from_center, _ = mcp.find_costs([center_idx])
    costs_from_center[np.isinf(costs_from_center)] = -1 # Ignore background
    pole_a = tuple(np.argwhere(costs_from_center == costs_from_center.max())[0])
    
    # Step 4: Find the furthest point from Pole A -> Pole B
    mcp = MCP_Geometric(cost_volume) # Reset MCP
    # FIX: Catch the returned arrays from find_costs
    costs_from_a, _ = mcp.find_costs([pole_a])
    costs_from_a[np.isinf(costs_from_a)] = -1
    pole_b = tuple(np.argwhere(costs_from_a == costs_from_a.max())[0])
    
    return pole_a, pole_b

def extract_centerline_anisotropic(binary_tube, power=2.0, spacing=(1.0, 1.0, 1.0)):
   
    pole_a, pole_b = get_geodesic_poles(binary_tube)

    dist = distance_transform_edt(binary_tube, sampling=spacing)
    max_dist = dist.max()

    cost_volume = np.where(
        binary_tube,
        (max_dist - dist + 1.0) ** power,
        np.inf,
    )

    mcp = MCP_Geometric(cost_volume, sampling=spacing)
    costs, traceback = mcp.find_costs([pole_a], [pole_b])

    path_coords = np.asarray(mcp.traceback(pole_b))

    physical_diffs = np.diff(path_coords, axis=0) * np.asarray(spacing)
    exact_length = np.linalg.norm(physical_diffs, axis=1).sum()

    return path_coords, exact_length

def create_centerline_mask(original_tube, path_coords):
    """
    Converts the 1D path coordinates back into a 3D binary mask 
    of the exact same shape as the input tube.
    """
    # Initialize an array of False with the same shape as the input
    centerline_mask = np.zeros_like(original_tube, dtype=bool)
    
    # Isolate the Z, Y, and X coordinate arrays
    # Note: MCP_Geometric returns coordinates in (Z, Y, X) order
    z_coords = path_coords[:, 0]
    y_coords = path_coords[:, 1]
    x_coords = path_coords[:, 2]
    
    # Set the voxels along the path to True
    centerline_mask[z_coords, y_coords, x_coords] = True
    
    return centerline_mask

def classify_image(image, labels, classifier_path):
    clf = joblib.load(classifier_path)
    result, class_names = clf.predict(image, labels)
    return result

class SkeletonCurve(TypedDict):
    length: float
    sampled_coords: NDArray[np.floating]
    smoothed_coords: NDArray[np.floating]

def calc_skeleton_curve(
        
    coords: NDArray[np.floating],
    sampling_dist: float,
    savgol_window=3,
    savgol_polyorder=1,
) -> SkeletonCurve:
    """
    Sample from a smooth curve created from coordinates derived from a skeletonization.

    First Savitzky-Golay filtering is applied, then the filtered coordinates are used to
    create piece-wise quadratic bezier curves.

    The `coords` should already be converted to real world coordinates to account for
    anisotropy in the data. The `sampling_dist` in real world units.

    Parameters
    ----------
    coords : NDArray[np.floating]
        Coordinates derived from the skeletonization of a volume. They should be in 
        real-world units to account for anisotropy. The coordinates should also be 
        ordered from the start of a path to the end of a path.
    sampling_dist : float
        The real-world (approximate) distance between the returned sampled points. It is
        likely desirable for the sampling distance to be smaller than the diagonal of a
        pixel, for sub pixel sampling.
    savgol_window : int, default=3
        The window size for Savitzky-Golay filtering.
    savgol_polyorder : int, default=1
        The polynomial order for Savitzky-Golay filtering.

    Returns
    -------
    SkeletonCurve
        A dictionary with the keys "length", "sampled_coords", and "smoothed_coords" for
        the length of the resulting curve, coordinates sampled on the curve, and the
        smoothed output of the intermediate Savitzky-Golay filtering step, respectively.
    """
    smoothed = savgol_filter(coords, savgol_window, savgol_polyorder, axis=0).astype(np.float32)
    b_curve = PiecewiseBezier(list(smoothed))

    length = b_curve.length()
    n_samples = int(np.ceil(length/sampling_dist))
    sampling_points = np.linspace(0, 1, n_samples)

    sampled_coords = b_curve(sampling_points)
    return {
        "length": length,
        "sampled_coords": sampled_coords,
        "smoothed_coords": smoothed
    }

def make_plot(df):

    plt.style.use("IAF_style.mplstyle")

    spline_lengths = df['spline_length'].tolist()
    branch_lengths_euc = df['dist_euc'].tolist()
    branch_lengths_path = df['dist_path'].tolist()

    metric_names = ['spline_length', 'dist_euc', 'dist_path']
    metric_lists = [spline_lengths, branch_lengths_euc, branch_lengths_path]



    means = [np.mean(vals) if len(vals) > 0 else 0 for vals in metric_lists]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(metric_names))
    ax.bar(x, means, width=0.6, alpha=0.2, linewidth=0.8)

    n_quantified = len(df)
    legend_label = f'n quantified = {n_quantified}'

    for i, vals in enumerate(metric_lists):
        if len(vals) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.8, linewidth=0.2, label=legend_label if i == 0 else None)

    if n_quantified > 0:
        ax.legend(title='Data', loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Length')
    ax.set_title('Average cilia measurements')
    plt.tight_layout()
    return plt#plt.show()
