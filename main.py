from pathlib import Path

import napari
from qtpy import QtWidgets
from biapy import BiaPy, build_config
import skimage
import numpy as np
from skan import Skeleton, summarize

#from functions import size_filter_masks, size_filter_masks, remove_circular_masks, classify_image, extract_cube_padded
from functions import *


def extract_filename(path: str) -> str:
    """Return the filename component of a path."""
    if not path:
        return ""
    return Path(path).name


def get_layer_name(layer) -> str:
    """Return the name assigned to a napari layer."""
    return layer.name


def _default_results_path(image_path: str) -> str:
    if not image_path:
        return ""
    return str(Path(image_path).expanduser().parent / "results")

def segment_image(cilia_im, model_checkpoint_path: str):
    print("segment_image called with the following parameters:")
    print(f"model_checkpoint_path: {model_checkpoint_path}")

    biapy = BiaPy.load_workflow_from_model(model_checkpoint_path, name="infer", gpu="0")

    print('model_loaded, beginning inference')
    mask = biapy.predict(cilia_im, verbose=False)
    print(mask.shape)
    # for file in files:
    print('inference complete')
    mask = np.squeeze(mask)
    return mask


class PipelineWidget(QtWidgets.QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._output_auto = True

        layout = QtWidgets.QVBoxLayout(self)

        image_row = self._make_path_row("Image", self._pick_image)
        model_row = self._make_path_row("Model checkpoint", self._pick_model)
        classifier_row = self._make_path_row("Classifier file", self._pick_classifier)
        output_row = self._make_path_row("Output path", self._pick_output, is_output=True)

        image_row.findChildren(QtWidgets.QLineEdit)[0].setText(
            "/facility/imganfac/neurogenomics/taverna/scordato_cilia_v2/test_data/czi/E10_ConA_Arl13b_NeuronalCilia_zoom1.8.czi"
        )
        model_row.findChildren(QtWidgets.QLineEdit)[0].setText(
            "/facility/imganfac/neurogenomics/taverna/scordato_cilia_v2/model_training_v2/cilia_zoom_semantic_new/checkpoints/cilia_zoom_semantic_new_1-checkpoint-best.pth"
        )
        classifier_row.findChildren(QtWidgets.QLineEdit)[0].setText(
            "/facility/imganfac/neurogenomics/taverna/scordato/classifiers/mouse_zoom.joblib"
        )

        layout.addWidget(image_row)
        layout.addWidget(model_row)
        layout.addWidget(classifier_row)
        layout.addWidget(output_row)

        numeric_group = QtWidgets.QGroupBox("Parameters")
        numeric_group.setCheckable(True)
        numeric_group.setChecked(False)
        self.numeric_group = numeric_group
        numeric_layout = QtWidgets.QFormLayout(numeric_group)

        self.cilia_channel = QtWidgets.QSpinBox()
        self.cilia_channel.setRange(0, 10**6)
        self.cilia_channel.setValue(0)

        self.binary_threshold = QtWidgets.QDoubleSpinBox()
        self.binary_threshold.setRange(0.0, 1.0)
        self.binary_threshold.setSingleStep(0.01)
        self.binary_threshold.setDecimals(3)
        self.binary_threshold.setValue(0.8)

        self.min_seg_size = QtWidgets.QSpinBox()
        self.min_seg_size.setRange(0, 10**9)
        self.min_seg_size.setValue(50)

        self.max_cilia_length = QtWidgets.QSpinBox()
        self.max_cilia_length.setRange(0, 10**9)
        self.max_cilia_length.setValue(4)

        self.max_circularity = QtWidgets.QDoubleSpinBox()
        self.max_circularity.setRange(0.0, 1.0)
        self.max_circularity.setSingleStep(0.01)
        self.max_circularity.setDecimals(3)
        self.max_circularity.setValue(0.95)

        self.max_axis_ratio = QtWidgets.QDoubleSpinBox()
        self.max_axis_ratio.setRange(0.0, 1.0)
        self.max_axis_ratio.setSingleStep(0.01)
        self.max_axis_ratio.setDecimals(3)
        self.max_axis_ratio.setValue(0.75)

        self.scale_xy = QtWidgets.QDoubleSpinBox()
        self.scale_xy.setRange(0.0, 1000.0)
        self.scale_xy.setSingleStep(0.001)
        self.scale_xy.setDecimals(6)

        self.scale_z = QtWidgets.QDoubleSpinBox()
        self.scale_z.setRange(0.0, 1000.0)
        self.scale_z.setSingleStep(0.001)
        self.scale_z.setDecimals(6)

        self.sliding_window = QtWidgets.QSpinBox()
        self.sliding_window.setRange(1, 10)
        self.sliding_window.setValue(5)

        self.savgol_polyorder = QtWidgets.QSpinBox()
        self.savgol_polyorder.setRange(0, 6)
        self.savgol_polyorder.setValue(2)

        numeric_layout.addRow("Cilia channel", self.cilia_channel)
        numeric_layout.addRow("Binary threshold", self.binary_threshold)
        numeric_layout.addRow("Min segmentation size", self.min_seg_size)
        numeric_layout.addRow("Max cilia length", self.max_cilia_length)
        numeric_layout.addRow("Max circularity", self.max_circularity)
        numeric_layout.addRow("Max axis ratio", self.max_axis_ratio)
        numeric_layout.addRow("Scale XY", self.scale_xy)
        numeric_layout.addRow("Scale Z", self.scale_z)
        numeric_layout.addRow("Sliding window", self.sliding_window)
        numeric_layout.addRow("Savgol polyorder", self.savgol_polyorder)
        layout.addWidget(numeric_group)

        numeric_group.toggled.connect(self._toggle_parameters_visibility)
        self._toggle_parameters_visibility(numeric_group.isChecked())

        self.load_images_button = QtWidgets.QPushButton("load images")
        self.load_images_button.clicked.connect(self.load_images)
        layout.addWidget(self.load_images_button)

        self.run_segmentation_button = QtWidgets.QPushButton("run segmentation")
        self.run_segmentation_button.clicked.connect(self.run_segmentation)
        layout.addWidget(self.run_segmentation_button)

        self.run_postprocessing_button = QtWidgets.QPushButton("run postprocessing")
        self.run_postprocessing_button.clicked.connect(self.run_postprocessing)
        layout.addWidget(self.run_postprocessing_button)

        self.run_quantification_button = QtWidgets.QPushButton("run quantification")
        self.run_quantification_button.clicked.connect(self.run_quantification)
        layout.addWidget(self.run_quantification_button)

        self.run_entire_pipeline_button = QtWidgets.QPushButton("run entire pipeline")
        self.run_entire_pipeline_button.clicked.connect(self.run_entire_pipeline)
        layout.addWidget(self.run_entire_pipeline_button)

        self.batch_model_button = QtWidgets.QPushButton("run batch mode")
        self.batch_model_button.clicked.connect(self.run_batch_mode)
        layout.addWidget(self.batch_model_button)

        layout.addStretch(1)

    def _toggle_parameters_visibility(self, checked: bool):
        for child in self.numeric_group.findChildren(QtWidgets.QWidget):
            child.setVisible(checked)

    def _make_path_row(self, label: str, browse_handler, is_output: bool = False):
        container = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)

        line_edit = QtWidgets.QLineEdit()
        line_edit.setPlaceholderText("Select a file" if not is_output else "Select an output folder")
        if is_output:
            line_edit.textEdited.connect(self._mark_output_manual)
        else:
            line_edit.textChanged.connect(self._sync_output_path)

        button = QtWidgets.QPushButton("Browse")
        button.clicked.connect(browse_handler)

        row_layout.addWidget(QtWidgets.QLabel(label))
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(button)

        if label == "Image":
            self.image_path = line_edit
        elif label == "Model checkpoint":
            self.model_path = line_edit
        elif label == "Classifier file":
            self.classifier_path = line_edit
        else:
            self.output_path = line_edit

        return container

    def _pick_file(self, target_line_edit: QtWidgets.QLineEdit, caption: str):
        start_dir = str(Path(target_line_edit.text()).expanduser().parent) if target_line_edit.text() else ""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, start_dir)
        if file_path:
            target_line_edit.setText(file_path)

    def _pick_directory(self):
        start_dir = self.output_path.text() or self.image_path.text()
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if directory:
            self._output_auto = False
            self.output_path.setText(directory)

    def _pick_image(self):
        self._pick_file(self.image_path, "Select image")

    def _pick_model(self):
        self._pick_file(self.model_path, "Select model checkpoint")

    def _pick_classifier(self):
        self._pick_file(self.classifier_path, "Select classifier file")

    def _pick_output(self):
        self._pick_directory()

    def _mark_output_manual(self):
        self._output_auto = False

    def _sync_output_path(self, _text: str):
        if not self._output_auto and self.output_path.text():
            return
        image_path = self.image_path.text().strip()
        if image_path:
            self._output_auto = True
            self.output_path.setText(_default_results_path(image_path))

    def _print_current_settings(self):
        image_path = self.image_path.text().strip()
        model_checkpoint_path = self.model_path.text().strip()
        classifier_file_path = self.classifier_path.text().strip()
        output_path = self.output_path.text().strip()

        binary_threshold = self.binary_threshold.value()
        min_segmentation_size = self.min_seg_size.value()
        max_cilia_length = self.max_cilia_length.value()
        max_circularity = self.max_circularity.value()

        print("pipeline run")
        print(f"image_path: {image_path}")
        print(f"model_checkpoint_path: {model_checkpoint_path}")
        print(f"classifier_file_path: {classifier_file_path}")
        print(f"output_path: {output_path}")
        print(f"binary_threshold: {binary_threshold}")
        print(f"min_segmentation_size: {min_segmentation_size}")
        print(f"max_cilia_length: {max_cilia_length}")
        print(f"max_circularity: {max_circularity}")

    def run_segmentation(self):
        print("run segmentation")
        model_checkpoint_path = self.model_path.text().strip()

        if "cilia_image" not in self.viewer.layers:
            QtWidgets.QMessageBox.warning(self, "Missing image", "please load images first")
            return

        cilia_image = self.viewer.layers["cilia_image"].data
        mask = segment_image(cilia_image, model_checkpoint_path)
        self.viewer.add_image(mask, name="probability_map")
        
    def load_images(self):
        print("load images")
        image_path = self.image_path.text().strip()
        if not image_path:
            QtWidgets.QMessageBox.warning(self, "Missing image", "please select an image first")
            return

        im, scale = open_and_parse_cilia_im(image_path)
        cilia_image = extract_cilia_channel(im, self.cilia_channel.value())
        #cilia_image = skimage.io.imread(image_path)
        if "cilia_image" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["cilia_image"])
        self.viewer.add_image(cilia_image, name="cilia_image")

        self.scale_z.setValue(scale[0])
        self.scale_xy.setValue(scale[1])

    def load_images_batch(self, image_path):
        print("load images")
    
        im, scale = open_and_parse_cilia_im(image_path)
        cilia_image = extract_cilia_channel(im, self.cilia_channel.value())
        #cilia_image = skimage.io.imread(image_path)
        self.viewer.layers.clear()
        if "cilia_image" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["cilia_image"])
        self.viewer.add_image(cilia_image, name="cilia_image")

        self.scale_z.setValue(scale[0])
        self.scale_xy.setValue(scale[1])
        

    def run_postprocessing(self):
        print("run postprocessing")
        binary_threshold = self.binary_threshold.value()
        min_segmentation_size = self.min_seg_size.value()
        max_circularity = self.max_circularity.value()
        max_axis_ratio = self.max_axis_ratio.value()
        elem_3D = skimage.morphology.footprint_rectangle((3,3,3))
        elem_2D = skimage.morphology.footprint_rectangle((3,3))
        
        if "cilia_image" not in self.viewer.layers:
            QtWidgets.QMessageBox.warning(self, "Missing image", "please load images first")
            return
        if "probability_map" not in self.viewer.layers:
            QtWidgets.QMessageBox.warning(self, "Missing probability_map", "please load images and segment first")
            return
        prob = self.viewer.layers["probability_map"].data
        cilia_im = self.viewer.layers["cilia_image"].data

        masks = prob >= binary_threshold
        self.viewer.add_labels(masks, name="raw_masks")

        masks_median = np.zeros_like(masks)
        for k in range(0, masks.shape[0]):
            masks_median[k,:,:] = skimage.filters.median(masks[k,:,:], footprint=elem_2D)
        #masks_median = skimage.filters.median(masks, footprint=elem_3D)
        masks_filled = skimage.morphology.remove_small_holes(masks_median > 0, max_size=16)

        labs = skimage.measure.label(masks_filled)
        #self.viewer.add_labels(labs, name="raw_labs")

        large_labs_removed, masks_removed = size_filter_masks(labs, min_segmentation_size, 5000)
        #self.viewer.add_labels(large_labs_removed, name="small_masks")

        labs_clean, masks_removed = remove_circular_masks(large_labs_removed, method='axes', max_circularity=max_circularity, max_axis=max_axis_ratio)
        #self.viewer.add_labels(labs_clean, name="circular_masks_removed")

        labs = skimage.measure.label(labs_clean)
        #self.viewer.add_labels(labs, name="cleaned_masks")
        result = classify_image(cilia_im, labs, self.classifier_path.text())
        result_temp = np.copy(result)
        self.viewer.add_labels(result_temp, name="classified")

        result[result == 1] = 0
        simple_masks_thresholded = labs * result
        self.viewer.add_labels(simple_masks_thresholded, name="final_masks")


    def run_quantification(self, imname=None, batchmode=False):

        scale_Z = self.scale_z.value()
        scale_XY = self.scale_xy.value()
        scale = (scale_Z, scale_XY, scale_XY)
        pixel_diag = np.linalg.norm(scale)

        print("run quantification")
        if "cilia_image" not in self.viewer.layers:
                QtWidgets.QMessageBox.warning(self, "Missing image", "please load images first")
                return
        if "final_masks" not in self.viewer.layers:
            QtWidgets.QMessageBox.warning(self, "Missing masks", "please load postprocess segmentation first")
            return
        simple_masks = self.viewer.layers["final_masks"].data
        cilia_im = self.viewer.layers["cilia_image"].data

        simple_skel = np.zeros(simple_masks.shape, dtype=np.uint16)
        props = skimage.measure.regionprops(simple_masks.astype(int))  
        for prop in props:
            label = prop.label
            cube_raw = extract_cube_padded(cilia_im, prop.bbox, prop.label, padding=0)
            cube = extract_cube_padded(simple_masks, prop.bbox, prop.label, padding=0)
            centerline_coords, length = extract_centerline_anisotropic(cube, power=2.0)
            center_mask = create_centerline_mask(cube, centerline_coords)
            if np.count_nonzero(center_mask) < 3:
                continue
            branch_data = summarize(Skeleton(center_mask), separator='-')
            if len(branch_data) == 1:
                simple_skel[prop.bbox[0]:prop.bbox[3], prop.bbox[1]:prop.bbox[4], prop.bbox[2]:prop.bbox[5]] += center_mask.astype(np.uint16) * prop.label
        self.viewer.add_labels(simple_skel, name="skeletons")


        props = skimage.measure.regionprops(simple_skel.astype(int))
        labels = []
        branch_lengths_euc = []
        branch_lengths_path = []
        spline_lengths = []
        sampled_coords = []
        sampled_coords_pixel_list = []
        for prop in props:
            cube_skel = extract_cube_padded(simple_skel, prop.bbox, label=prop.label, padding=0)
            
            skeleton = Skeleton(cube_skel,spacing=scale) 
            branch_data = summarize(Skeleton(cube_skel, spacing=scale), separator='-')
            if len(branch_data) > 1:
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
                savgol_window=self.sliding_window.value(),
                savgol_polyorder=self.savgol_polyorder.value(),
                )   
        
            sampled_coords_pixel = result["sampled_coords"] / np.array(scale)
            labels.append(prop.label)
            spline_lengths.append(np.round(result['length'], 2))
            branch_lengths_euc.append(np.round(branch_data['euclidean-distance'].values[0], 2))
            branch_lengths_path.append(np.round(branch_data['branch-distance'].values[0], 2))
            sampled_coords.append(result['sampled_coords'])
            sampled_coords_pixel_list.append(sampled_coords_pixel)
        df = pd.DataFrame(columns=['roi_label', 'spline_length', 'dist_euc', 'dist_path', 'sampled_coords'])
        df['roi_label'] = labels
        df['spline_length'] = spline_lengths
        df['dist_euc'] = branch_lengths_euc
        df['dist_path'] = branch_lengths_path
        #df['sampled_coords'] = sampled_coords
        #df['sampled_coords_pixel'] = sampled_coords_pixel_list
        #df.to_csv(os.path.join(OUTPUT_PATH, 'quantification.csv'), index=False)
        plot = make_plot(df)

        if imname is None:
            save_path = self.output_path.text().strip()
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, 'quantification.csv'), index=False)
            skimage.io.imsave(os.path.join(save_path, 'skeletons.tif'), self.viewer.layers['skeletons'].data.astype(np.uint16), check_contrast=False)
            skimage.io.imsave(os.path.join(save_path, 'final_masks.tif'), self.viewer.layers['final_masks'].data.astype(np.uint16), check_contrast=False)
            plot.savefig(os.path.join(save_path, 'quantification_plot.png'))
        else:
            save_path = os.path.join(self.output_path.text().strip(), extract_filename(imname))
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, 'quantification.csv'), index=False)
            skimage.io.imsave(os.path.join(save_path, 'skeletons.tif'), self.viewer.layers['skeletons'].data.astype(np.uint16), check_contrast=False)
            skimage.io.imsave(os.path.join(save_path, 'final_masks.tif'), self.viewer.layers['final_masks'].data.astype(np.uint16), check_contrast=False)
            plot.savefig(os.path.join(save_path, 'quantification_plot.png'))
        
        if not batchmode:
            plot.show()

        all_sampled_points = np.vstack(sampled_coords_pixel_list) if sampled_coords_pixel_list else np.empty((0, 3))
        self.viewer.add_points(all_sampled_points, name='sampled_coords_pixel', size=2, face_color='cyan')

    

    def run_entire_pipeline(self):
        self._print_current_settings()
        self.load_images()
        self.run_segmentation()
        self.run_postprocessing()
        self.run_quantification()
        self.viewer.layers['probability_map'].visible = False
        self.viewer.layers['raw_masks'].visible = False
        self.viewer.layers['skeletons'].visible = False

    def run_batch_mode(self):
        start_dir = (
            str(Path(self.image_path.text()).expanduser().parent)
            if self.image_path.text().strip()
            else ""
        )
        batch_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder to run batch mode in", start_dir
        )
        if not batch_folder:
            print("batch mode cancelled: no folder selected")
            return
        print('Running batch mode on folder:', batch_folder)
        self._print_current_settings()

        file_list = os.listdir(batch_folder)
        file_list = [f for f in file_list if f.endswith(('.tif', '.tiff', '.czi', '.nd2', '.lif'))]
        for file in file_list:
            image_path = os.path.join(batch_folder, file)
            self.load_images_batch(image_path)
            self.run_segmentation()
            self.run_postprocessing()
            self.run_quantification(file, batchmode=True)
            self.viewer.layers['probability_map'].visible = False
            self.viewer.layers['raw_masks'].visible = False
            self.viewer.layers['skeletons'].visible = False
            

def main():
    viewer = napari.Viewer()
    widget = PipelineWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Pipeline", area="right")
    napari.run()


if __name__ == "__main__":
    main()
