"""
object_classifier.py  
--------------------
Train and apply a random-forest classifier on objects in a 3D label image.

Workflow
--------
1. Extract per-object features from scikit-image regionprops (all scalar 3D
   properties are attempted; any that fail are silently dropped).
2. Build a training set from a user-supplied class dictionary.
3. Fit a scikit-learn RandomForestClassifier.
4. Predict classes for every object and return a colour-coded label image.

Example
-------
    from object_classifier import ObjectClassifier

    classes = {
        'neuron': [1, 3, 7],
        'glia':   [2, 5],
    }

    clf = ObjectClassifier()
    clf.fit(raw, labels, classes)

    result, class_names = clf.predict(raw, labels)
    # result[i, j, k] == 1  →  class_names[0]
    # result[i, j, k] == 2  →  class_names[1]
    # result[i, j, k] == 0  →  background
"""

import warnings
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Scalar regionprops properties to attempt (3D-compatible)
# Properties that require 2D (e.g. perimeter, eccentricity) are excluded.
# ---------------------------------------------------------------------------
_MORPHOLOGY_PROPS = [
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "solidity",
]

_INTENSITY_PROPS = [
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "intensity_std",
]


def extract_features(labels: np.ndarray, raw: np.ndarray) -> pd.DataFrame:
    """
    Extract per-object features from a label image using regionprops.

    Each regionprops property is attempted independently; properties that raise
    an exception for any object are dropped from the output entirely.
    The inertia-tensor eigenvalues (3 values per object in 3D) are expanded
    into separate columns ``inertia_ev_0``, ``inertia_ev_1``, ``inertia_ev_2``.
    An ``axis_ratio`` column (minor / major axis length) is also added.

    Parameters
    ----------
    labels : ndarray, shape (Z, Y, X)
        Integer label image; 0 is background.
    raw : ndarray, shape (Z, Y, X)
        Intensity image used for intensity-based features.

    Returns
    -------
    df : pd.DataFrame
        One row per object.  The ``label`` column holds the original label ID.
    """
    scalar_props = tuple(_MORPHOLOGY_PROPS) + tuple(_INTENSITY_PROPS)
    all_props = ("label",) + scalar_props + ("inertia_tensor_eigvals",)

    data: dict[str, np.ndarray] = {}

    # Fast path: try to compute every property in a single regionprops_table
    # call (one pass over the image).  Fall back to property-by-property
    # extraction only if any property causes the batch call to fail.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = regionprops_table(labels, intensity_image=raw, properties=all_props)
        data["label"] = table["label"]
        n_objects = len(data["label"])
        for prop in scalar_props:
            if prop in table:
                values = table[prop]
                if len(values) == n_objects and np.issubdtype(values.dtype, np.number):
                    data[prop] = values
        for i in range(3):
            col = f"inertia_tensor_eigvals-{i}"
            if col in table and len(table[col]) == n_objects:
                data[f"inertia_ev_{i}"] = table[col]

    except Exception:
        # Slow fallback: attempt each property independently so that a single
        # failing property does not block the others.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = regionprops_table(labels, intensity_image=raw, properties=("label",))
        data["label"] = base["label"]
        n_objects = len(data["label"])

        for prop in scalar_props:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    table = regionprops_table(
                        labels, intensity_image=raw, properties=("label", prop)
                    )
                values = table[prop]
                if len(values) == n_objects and np.issubdtype(values.dtype, np.number):
                    data[prop] = values
            except Exception:
                pass

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                table = regionprops_table(
                    labels, intensity_image=raw,
                    properties=("label", "inertia_tensor_eigvals"),
                )
            for i in range(3):
                col = f"inertia_tensor_eigvals-{i}"
                if col in table and len(table[col]) == n_objects:
                    data[f"inertia_ev_{i}"] = table[col]
        except Exception:
            pass

    df = pd.DataFrame(data)

    # Derived feature: axis ratio (minor / major length)
    if "axis_major_length" in df.columns and "axis_minor_length" in df.columns:
        major = df["axis_major_length"].replace(0, np.nan)
        df["axis_ratio"] = df["axis_minor_length"] / major

    # Replace inf/-inf and NaN with the column mean.
    # Infinities arise from divisions (e.g. axis_ratio when major_length == 0)
    # or from degenerate regionprops computations.
    feature_cols = [c for c in df.columns if c != "label"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    return df


class ObjectClassifier:
    """
    Random-forest classifier for objects in a 3D label image.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the random forest.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.forest = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.class_names: list[str] | None = None
        self.feature_columns: list[str] | None = None
        self.feature_means_: dict[str, float] = {}

    def fit(
        self,
        raw: np.ndarray,
        labels: np.ndarray,
        classes: dict[str, list[int]],
        features: list[str] | None = None,
        crop_to_training: bool = True,
    ) -> "ObjectClassifier":
        """
        Extract features and train the classifier on labelled examples.

        Parameters
        ----------
        raw : ndarray
            Intensity image.
        labels : ndarray
            Integer label image (0 = background).
        classes : dict
            ``{'class_name': [label_id, ...], ...}``
            Only objects whose label ID appears in this dict are used for
            training; all others are classified at prediction time.
        features : list of str, optional
            Subset of feature names to use for training and prediction.
            If ``None`` (default) all extracted features are used.
            Pass the ``'feature'`` column from ``feature_importances()``
            (e.g. the top-10 entries) to restrict training to those features
            and speed up both fitting and inference.
        crop_to_training : bool, optional
            If ``True`` (default), feature extraction is restricted to the
            tight bounding box containing only the training objects, which
            is much faster on large images.  Set to ``False`` to run
            regionprops over the full image instead.

        Returns
        -------
        self
        """
        # Map every label ID to its class name
        label_to_class: dict[int, str] = {}
        for class_name, label_ids in classes.items():
            for lid in label_ids:
                label_to_class[lid] = class_name

        if crop_to_training:
            # Speed up feature extraction by restricting to the bounding box
            # containing only the training objects.  This avoids running
            # regionprops over the full image (which may contain thousands of
            # irrelevant objects).
            all_training_ids = list(label_to_class.keys())
            training_voxel_mask = np.isin(labels, all_training_ids)
            if training_voxel_mask.any():
                coords = np.argwhere(training_voxel_mask)
                z0, y0, x0 = coords.min(axis=0)
                z1, y1, x1 = coords.max(axis=0) + 1
                labels_fit = np.where(training_voxel_mask, labels, 0)[z0:z1, y0:y1, x0:x1]
                raw_fit = raw[z0:z1, y0:y1, x0:x1]
            else:
                labels_fit = np.where(training_voxel_mask, labels, 0)
                raw_fit = raw
        else:
            labels_fit = labels
            raw_fit = raw

        features_df = extract_features(labels_fit, raw_fit)

        # Filter to training objects only
        train_mask = features_df["label"].isin(label_to_class)
        train_df = features_df[train_mask].copy()
        train_df["class"] = train_df["label"].map(label_to_class)

        if train_df.empty:
            raise ValueError(
                "None of the label IDs in `classes` were found in the label image."
            )

        # Feature matrix and target vector
        self.class_names = sorted(classes.keys())
        all_feature_columns = [
            c for c in train_df.columns if c not in ("label", "class")
        ]
        if features is not None:
            missing = set(features) - set(all_feature_columns)
            if missing:
                raise ValueError(
                    f"Requested features not found in extracted features: {missing}"
                )
            self.feature_columns = [f for f in features if f in all_feature_columns]
        else:
            self.feature_columns = all_feature_columns

        X_df = train_df[self.feature_columns]
        self.feature_means_ = X_df.mean().to_dict()
        X = X_df.values
        y = train_df["class"].values

        print(f"Training on {len(train_df)} objects, {len(self.feature_columns)} features, "
              f"{len(self.class_names)} classes: {self.class_names}")
        self.forest.fit(X, y)
        print("Done.")
        return self

    def predict(
        self,
        raw: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Classify every object in the label image.

        Parameters
        ----------
        raw : ndarray
            Intensity image.
        labels : ndarray
            Integer label image (0 = background).

        Returns
        -------
        result : ndarray (same shape as labels)
            Each object's voxels are filled with its class index (1-based).
            Background = 0.  Index i corresponds to ``class_names[i - 1]``.
        class_names : list of str
            Ordered class names.  ``class_names[i - 1]`` is the class for
            voxels with value ``i`` in ``result``.
        """
        if self.feature_columns is None:
            raise RuntimeError("Call fit() before predict().")

        features_df = extract_features(labels, raw)

        # Use only the features seen during training.
        # Fill any features that were present at training time but couldn't be
        # computed on this image (e.g. due to image properties or object count)
        # with the mean value recorded during fit().
        missing = set(self.feature_columns) - set(features_df.columns)
        if missing:
            print(f"Warning: {len(missing)} feature(s) missing at inference time, "
                  f"filling with training means: {missing}")
            for col in missing:
                features_df[col] = self.feature_means_.get(col, 0.0)

        X = features_df[self.feature_columns].values
        predicted_classes = self.forest.predict(X)

        # Map class names → 1-based integer index
        class_to_idx = {name: i + 1 for i, name in enumerate(self.class_names)}

        # Build a lookup table (label_id → class_index) and apply it with a
        # single vectorised index operation instead of one boolean-mask scan
        # per object.  This is O(image_size) rather than O(n_objects × size).
        max_label = int(labels.max())
        lut = np.zeros(max_label + 1, dtype=np.result_type(len(self.class_names), np.uint8))
        class_indices = np.array(
            [class_to_idx[c] for c in predicted_classes], dtype=lut.dtype
        )
        lut[features_df["label"].values] = class_indices
        result = lut[labels]

        return result, self.class_names

    def feature_importances(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances, sorted descending.

        Returns
        -------
        pd.DataFrame with columns ``feature`` and ``importance``.
        """
        if self.feature_columns is None:
            raise RuntimeError("Call fit() before feature_importances().")
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.forest.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
