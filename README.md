# Cilia segmentation and quantification pipeline

A [napari](https://napari.org) plugin-style widget (and a few standalone scripts) for
segmenting primary cilia in 3D fluorescence stacks and measuring their length.

The pipeline:

1. **Load** a 3D image (`.czi`, `.lif`, ...) and extract the cilia channel, together with the voxel size from the metadata.
2. **Segment** with a [BiaPy](https://biapy.readthedocs.io) semantic-segmentation model, producing a probability map.
3. **Post-process** the probability map into clean, labelled cilia masks (optionally filtered by a random-forest classifier).
4. **Quantify** each cilium: extract a centerline skeleton, fit a smoothed Bezier curve and report lengths.

## Installation

Requires Python >= 3.13 (see [.python-version](.python-version)) and [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repository root: create .venv and install everything from pyproject.toml / uv.lock
uv sync
```

`uv sync` creates a `.venv` with the pinned dependencies (BiaPy, bioio, napari, scikit-image tools,
skan, torch, scikit-learn, bezier, ...). `bezier` is installed from the
`nobias-fht/bioimage-toolbox` git repository, so network access to GitHub is needed.

Run commands inside the environment with `uv run`, e.g. `uv run python main.py`, or activate it with
`source .venv/bin/activate`.

## Usage

### napari GUI (`main.py`)

```bash
uv run python main.py
```

This opens napari with a **Pipeline** dock widget on the right.

**Inputs**

| Field | Description |
| --- | --- |
| Image | Image to analyse (`.czi` / `.lif`). |
| Model checkpoint | BiaPy checkpoint (`.pth`) used for segmentation. |
| Classifier file | Trained `ObjectClassifier` saved with joblib (`.joblib`). |
| Output path | Folder for results. Defaults to a `results` folder next to the image. |

**Parameters** (collapsed by default, tick the *Parameters* box to expand)

| Parameter | Default | Meaning |
| --- | --- | --- |
| Use classifier | checked | Filter masks with the classifier. When unchecked, classification is skipped and all masks that pass the size/shape filters are kept. |
| Cilia channel | 0 | Channel index containing the cilia. |
| Binary threshold | 0.8 | Probability-map threshold used to binarise the segmentation. |
| Min segmentation size | 50 | Minimum object size (voxels); the maximum is fixed at 5000. |
| Max cilia length | 4 | Shown in the settings printout. |
| Max circularity | 0.95 | Reserved for circularity filtering. |
| Max axis ratio | 0.75 | Objects whose minor/major axis ratio (Z max-projection) is above this are removed as round blobs. |
| Scale XY / Scale Z | from metadata | Voxel size; filled automatically when an image is loaded. |
| Sliding window | 5 | Savitzky-Golay window used to smooth skeleton coordinates. |
| Savgol polyorder | 2 | Savitzky-Golay polynomial order. |

**Buttons**

| Button | What it does | Layers / outputs |
| --- | --- | --- |
| load images | Reads the image, extracts the cilia channel and voxel size. | `cilia_image` |
| run segmentation | Runs BiaPy inference on `cilia_image`. | `probability_map` |
| run postprocessing | Thresholds, median-filters (per slice), fills holes, labels, removes objects by size and roundness, then (if *Use classifier* is checked) classifies objects and drops class 1. | `raw_masks`, `classified` (only with classifier), `final_masks` |
| run quantification | Skeletonises each mask (keeping unbranched ones), fits a smooth curve and measures lengths. | `skeletons`, `sampled_coords_pixel`, files in the output path |
| run entire pipeline | load images, segmentation, postprocessing and quantification in one go. | all of the above |
| run batch mode | Prompts for a folder and runs the full pipeline on every `.tif/.tiff/.czi/.nd2/.lif` in it. | one sub-folder per image in the output path |

The steps must be run in order; the widget warns if a required layer is missing.

### Outputs

Written to the output path (in batch mode, to `<output>/<image filename>/`):

- `quantification.csv` - one row per cilium: `roi_label`, `spline_length`, `dist_euc` (Euclidean end-to-end distance), `dist_path` (path length along the skeleton). Lengths are in the physical units of the image metadata.
- `skeletons.tif` - labelled centerlines.
- `final_masks.tif` - final labelled cilia masks.
- `quantification_plot.png` - summary plot of the three length measures.

### Standalone script (`pipeline.py`)

[pipeline.py](pipeline.py) is the non-interactive version of the same workflow. Paths, thresholds and
the image index are hard-coded constants at the top of the file; edit them and run
`uv run python pipeline.py`. It always applies the classifier (`mouse_zoom.joblib` for files with
"zoom" in the name, otherwise `mouse_nozoom.joblib`).

## Repository layout

| File | Purpose |
| --- | --- |
| [main.py](main.py) | napari GUI widget (entry point). |
| [functions.py](functions.py) | Shared processing functions: image loading, size/roundness filters, centerline extraction, classification, curve fitting, plotting. |
| [object_classifier.py](object_classifier.py) | `ObjectClassifier`: random forest on per-object `regionprops` features of a 3D label image. |
| [pipeline.py](pipeline.py) | Script version of the pipeline. |
| [train_nn.py](train_nn.py) | Trains the BiaPy segmentation model from a YAML config. |
| [step3_extract_skeletons_classifier.py](step3_extract_skeletons_classifier.py) | Earlier skeleton-extraction script. |
| [IAF_style.mplstyle](IAF_style.mplstyle) | Matplotlib style used for the quantification plot. |

## Training

- **Segmentation model:** edit the YAML and output paths in [train_nn.py](train_nn.py), then run `uv run python train_nn.py`. The resulting `checkpoints/*-checkpoint-best.pth` is the *Model checkpoint* used above.
- **Classifier:** fit an `ObjectClassifier` on a raw image and its label image, giving the label ids that belong to each class, then save it with `joblib.dump`. See the docstring of [object_classifier.py](object_classifier.py) for an example. In the pipeline, predicted class 1 is treated as "reject" and removed.

## Notes

- Plotting loads `IAF_style.mplstyle` by relative path, so run the GUI from this directory.
- Segmentation uses a GPU (`gpu="0"`) via BiaPy.
