
import os
import numpy as np
import colorsys
import math
import colorsys
import cv2 # OpenCV for reading image dimensions
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import yaml # For reading data.yaml if you have one
import random # To generate random colors



def load_yolo_seg_dicts(image_dir, label_dir, class_names_list):
    """
    Loads YOLO segmentation annotations and converts them into Detectron2's 
    standard dataset dictionary format.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing YOLO label files.
        class_names_list (list[str]): List of class names to determine num_classes
                                      and for potential validation.

    Returns:
        list[dict]: A list of dictionaries, one for each image.
    """
    dataset_dicts = []
    img_id_counter = 0 
    num_classes = len(class_names_list)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_extensions)]
   
    print(f"Processing {len(image_files)} images from {image_dir}...")

    for img_filename in image_files:
        record = {}
        image_path = os.path.join(image_dir, img_filename)

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path} (cv2.imread returned None), skipping.")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Warning: Could not read image {image_path} to get dimensions, skipping. Error: {e}")
            continue

        record["file_name"] = image_path
        record["image_id"] = img_id_counter # Could also use os.path.splitext(img_filename)[0]
        record["height"] = height
        record["width"] = width
        img_id_counter += 1

        label_filename_base = os.path.splitext(img_filename)[0]
        label_path = os.path.join(label_dir, label_filename_base + ".txt")

        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue

                    try:
                        class_id = int(parts[0])
                        # Validate class_id
                        if not (0 <= class_id < num_classes):
                            print(f"Warning: Invalid class_id {class_id} in {label_path} (line {line_idx+1}). "
                                  f"Expected 0 to {num_classes-1}. Skipping this annotation.")
                            continue
                        
                        # Normalized polygon coordinates (x1 y1 x2 y2 ... xn yn)
                        poly_normalized = [float(p) for p in parts[1:]]
                    except ValueError as e:
                        print(f"Warning: Error parsing line in {label_path} (line {line_idx+1}): '{line.strip()}'. Error: {e}. Skipping annotation.")
                        continue
                    
                    # Denormalize polygon coordinates
                    poly_absolute = []
                    if len(poly_normalized) % 2 != 0:
                        print(f"Warning: Odd number of polygon coordinates in {label_path} (line {line_idx+1}). Skipping annotation.")
                        continue

    for i in range(0, len(poly_normalized), 2):
                            x_norm = poly_normalized[i]
                            y_norm = poly_normalized[i+1]
                            poly_absolute.append(x_norm * width)
                            poly_absolute.append(y_norm * height)
                        
                            # Ensure polygon has at least 3 points (6 values)
                            if len(poly_absolute) < 6:
                                print(f"Warning: Invalid polygon (less than 3 points) in {label_path} for {img_filename} (line {line_idx+1}). Skipping annotation.")
                                continue

                            # Calculate bounding box from polygon
                            poly_np = np.array(poly_absolute).reshape(-1, 2)
                            min_x, min_y = np.min(poly_np, axis=0)
                            max_x, max_y = np.max(poly_np, axis=0)

                            obj = {
                                "bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "segmentation": [poly_absolute], # List of polygons
                                "category_id": class_id,
                                "iscrowd": 0 
                            }
                            annotations.append(obj)
            
    record["annotations"] = annotations
    dataset_dicts.append(record)
                            
    print(f"Finished processing. Loaded {len(dataset_dicts)} image records.")
    return dataset_dicts


def register_yolo_seg_dataset(dataset_name_prefix, dataset_root, class_names, splits=("train", "val")):
    """
    Registers YOLO segmentation datasets (for specified splits) with Detectron2
    and sets their metadata.

    Args:
        dataset_name_prefix (str): A prefix for the registered dataset names
                                   (e.g., "my_yolo_seg").
        dataset_root (str): The root directory of the YOLO dataset.
        class_names (list[str]): A list of class names for the dataset.
        splits (tuple[str]): A tuple of dataset splits to register 
                             (e.g., ("train", "val", "test")).
                             Assumes subdirectories like 'train/images', 'train/labels', etc.
    """
    # Generate random colors for each class for visualization
    thing_colors = [
    tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / len(class_names), 0.85, 0.85))
    for i in range(len(class_names))
]


    for d in splits:
        image_dir = os.path.join(dataset_root, d, "images")
        label_dir = os.path.join(dataset_root, d, "labels")
        dataset_name = f"{dataset_name_prefix}_{d}" 

    

        # Register the dataset
        # The lambda function captures the current values of image_dir, label_dir, and class_names
        # for when the loader function is actually called by Detectron2.
        DatasetCatalog.register(
            dataset_name,
            lambda idir=image_dir, ldir=label_dir, cnames=class_names: load_yolo_seg_dicts(idir, ldir, cnames)
        )

        # Set metadata for the registered dataset
        MetadataCatalog.get(dataset_name).set(
            thing_classes=class_names,
            thing_colors=thing_colors,
            image_root=image_dir, # Often useful for evaluators or visualizers
            label_root=label_dir, # Custom metadata, might be useful
            evaluator_type="coco"  # Assuming COCO-style evaluation
            # Add other metadata if needed
        )
        
        print(f"\nSuccessfully registered dataset: '{dataset_name}'")
        print(f"  Image directory: {image_dir}")
        print(f"  Label directory: {label_dir}")
        print(f"  Metadata 'thing_classes': {MetadataCatalog.get(dataset_name).thing_classes}")
        


def register_dataset():


    
        
    CLASS_NAMES = ["rca", "pda", "pborca"]

    DATASET_ROOT = "dataset_yolo/" 

    dataset_name_prefix = 'right_contrast_v1'

    splits=("train", "val")


    register_yolo_seg_dataset(
            dataset_name_prefix=dataset_name_prefix, 
            dataset_root=DATASET_ROOT,
            class_names=CLASS_NAMES,
            splits=splits  
        )