# AI Traffic Police 🚦

A computer vision system that helps traffic authorities monitor roads, detect incidents, and improve traffic management — built as part of the CSI-VITC / VIT Chennai Computer Vision project track.

## Project Overview

This project builds a CNN-based vehicle classification and detection pipeline in stages, progressing from basic vehicle-type classification (Part 1: Foundations) toward incident/violation detection (Part 2: Detection) and an accident-prevention dashboard (Part 3: Advanced).

Currently implemented: **Part 1 — vehicle type classification** (Bus, Motorcycle, car, truck) using a custom CNN trained in TensorFlow/Keras.

## Problem Statement

Manual traffic monitoring is labor-intensive and error-prone, especially at scale across multiple intersections and camera feeds. This project explores how computer vision can automate key traffic-authority tasks — starting with reliably classifying vehicle types from images, as a foundation for downstream tasks like emergency-vehicle detection, violation detection, and accident prevention.

## Installation Instructions

This project is built and run in **Google Colab**.

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd ai-traffic-police
   ```
2. Open the notebook (`.ipynb`) in Google Colab.
3. Install dependencies (if running outside Colab, which comes with most pre-installed):
   ```bash
   pip install tensorflow opencv-python pandas numpy scikit-learn matplotlib seaborn
   ```
4. Upload the dataset zip to Google Drive, then mount Drive in the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Update the `zip_file_path` variable to point to your dataset's location in Drive, then run all cells in order.

## Dataset Used

- **Source**: Roboflow-style vehicle image dataset ("Vehicles.zip"), pre-split into `train`, `test`, and `valid` folders.
- **Format**: Each split folder contains images plus a `_classes.csv` file mapping filenames to one-hot encoded class labels.
- **Classes**: Bus, Motorcycle, car, truck.
- **Size**: ~4,300 training images, ~150 validation images, ~90 test images (after filtering out multi-label and unlabeled images for single-label classification).

## Methodology

1. **Data loading & cleaning**: Parsed `_classes.csv` per split, dropped the placeholder "0"/background column, identified and excluded multi-label and no-label rows to build a clean single-label classification dataset.
2. **Preprocessing**: Images resized to a fixed size, normalized to [0, 1], and one-hot encoded labels (encoder fit on `train`, reused for `val`/`test` to keep label ordering consistent).
3. **Model architecture**: A custom CNN with:
   - Data augmentation layers (random flip, rotation, zoom, contrast)
   - Three convolutional blocks (32 → 64 → 128 filters) with BatchNormalization, MaxPooling, and Dropout
   - A dense classification head with L2 regularization and Dropout
4. **Training**: Trained with the Adam optimizer and categorical cross-entropy loss, using `EarlyStopping` (monitoring validation loss) to prevent overfitting.
5. **Evaluation**: Final performance measured on a held-out `test` set (never used during training or hyperparameter tuning) via accuracy and a confusion matrix.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Pandas / NumPy
- scikit-learn (OneHotEncoder)
- Matplotlib / Seaborn
- Google Colab + Google Drive

## Results

- **Test accuracy**: ~82% across four vehicle classes.
- **Per-class performance**: Strongest on Motorcycle (~93% recall); Bus and car show more confusion with each other and with truck, reflecting their visual similarity (shared box-like shapes, similar road-level framing).
- Training/validation accuracy and loss curves, along with the full confusion matrix, are included in the notebook.

## Challenges Faced

- **Large dataset upload**: The ~650MB dataset zip repeatedly failed or truncated when uploaded directly into a Colab session; resolved by uploading once to Google Drive and mounting Drive in Colab instead.
- **Nested/inconsistent folder structure**: The extracted zip contained an extra nested folder layer, requiring adjustments to file path construction.
- **Multi-label and unlabeled data**: A portion of the dataset had images with zero or multiple class labels; these were filtered out to keep the initial model scope to clean single-label classification.
- **Overfitting**: Initial training showed a large gap between training and validation accuracy/loss. Addressed through a combination of data augmentation, BatchNormalization, Dropout, L2 regularization, and early stopping — which meaningfully closed the gap and improved final test accuracy.
- **Slow Drive-mounted I/O**: Extracting directly from a Drive-mounted path was slow; resolved by copying the zip to Colab's local disk before extraction.

## Future Improvements

- **Transfer learning**: Swap the custom CNN for a pretrained backbone (e.g., MobileNetV2) to improve accuracy on visually similar classes (Bus, car, truck) given the relatively small dataset size.
- **True multi-label classification**: Extend the model to handle images containing multiple vehicle types, rather than filtering them out.
- **Object detection (Part 2)**: Move from whole-image classification to bounding-box-based detection (e.g., YOLOv8) to support emergency vehicle detection and basic traffic violation detection in multi-vehicle scenes.
- **Video/temporal reasoning**: Add vehicle tracking across frames to support violation detection (e.g., red-light running, wrong-lane driving) and accident detection.
- **Analytics dashboard, multi-camera support, heatmaps**: Stretch goals for a full traffic-monitoring system.


---

**Author**: Anirudh S.J
**Institution**: VIT Chennai — CSI-VITC Computer Vision Track
