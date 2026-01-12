# libraries

import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os, os.path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

notebooks_path = Path.cwd()
repo_path = notebooks_path.parent
data_path = repo_path / 'data' / 'HematomaTruetaBasal' 
md_path = repo_path / "metadata" / 'metadata.csv'


def get_indexes(dataset):
    """ The function splits train and test set and applied stratified cross validation to the train and validation set. 

    Args:
        dataset (_type_): _description_

    Returns:
        X_train (list): List of indexes of the train set.
        X_val (list): List of indexes of the validation set.
        X_test (list): List of indexes of the test set.
        y_train (list): List of labels of the train set.
        y_val (list): List of labels of the validation set.
        y_test (list): List of labels of the test set. 
    """
    # separate the dataset into train and test using stratified setting. 
    X_train_indexes, X_test, y_train_indexes , y_test = train_test_split(dataset.index, dataset.labels, test_size=0.2, random_state=42, stratify=dataset.labels)
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X_train_indexes, y_train_indexes):
        X_train, X_val = X_train_indexes[train_index], X_train_indexes[val_index]
        y_train, y_val = y_train_indexes[train_index], y_train_indexes[val_index]

    return X_train, X_val, X_test, y_train, y_val, y_test

def resize_or_pad_slice(slice_2d, target_size=(512, 512)):
    """
    Resize a 2D slice to target size using center cropping or zero padding.
    
    :param slice_2d: 2D numpy array (H, W)
    :param target_size: tuple of (height, width), default (512, 512)
    :return: resized/padded slice of shape target_size
    """
    h, w = slice_2d.shape
    target_h, target_w = target_size
    
    # Initialize output with zeros (padding value)
    output = np.zeros((target_h, target_w), dtype=slice_2d.dtype)
    
    # Calculate cropping/padding offsets
    # For height
    if h > target_h:
        # Crop: take center portion
        start_h = (h - target_h) // 2
        crop_h = slice_2d[start_h:start_h + target_h, :]
    else:
        # Pad: center the image
        start_h = (target_h - h) // 2
        crop_h = slice_2d
        
    # For width
    if w > target_w:
        # Crop: take center portion
        start_w = (w - target_w) // 2
        crop_w = crop_h[:, start_w:start_w + target_w]
    else:
        # Pad: center the image
        start_w = (target_w - w) // 2
        crop_w = crop_h
    
    # Place the cropped/original image in the center of the output
    actual_h, actual_w = crop_w.shape
    if h <= target_h and w <= target_w:
        # Both dimensions need padding
        output[start_h:start_h + actual_h, start_w:start_w + actual_w] = crop_w
    elif h > target_h and w <= target_w:
        # Height cropped, width padded
        output[:, start_w:start_w + actual_w] = crop_w
    elif h <= target_h and w > target_w:
        # Height padded, width cropped
        output[start_h:start_h + actual_h, :] = crop_w
    else:
        # Both dimensions cropped
        output = crop_w
    
    return output

def get_slices(loader, return_type=None): # only for the image slices not mask or label. 
    """ The function takes a dataloader and returns a list of slices of the given set pateints.
    The slices are in 2d image format (512,512) and they are ansembled in a list.

    Args:
        loader (data loader): data loader of the set of patients.

    Returns:
        list: image slices and the corresponding labels of the given set of patients.
              or image masks and the corresponding labels of the given set of patients.
    """
    # Brings all the slices of the test set patients in to together in 2d image format (512,512)
    image_slices = []
    labels = [] # labels per each slice
    masks = [] # masks per each slice
    for batch in loader:
        image, mask, label = batch
        for idx in range(image.size(1)): #image.size(1) is the number of slices in the image.
            image_slice = image[0, idx, :, :] # concatinating the slices of the same patient.
            mask_slice = mask[0, idx, :, :] 
            image_slices.append(image_slice)
            labels.append(label)  # for the same patient's slices, the label is the same.
            masks.append(mask_slice) # for the same patient's slices, the mask is the same.
    if return_type == 'image':
        return image_slices, labels
    elif return_type == 'mask':
        return masks, labels 
    
def add_dim_to_slices(slices):
    """
        Add channel dimension because data augmentations expect data in shape (C, H, W)
    Args:
        slices (tensor): tensor of slices
    Returns:
        tensor: tensor of slices with channel dimension (1, H, W)
    """
    slices_updated = []
    for i in slices:
        i = i[None, :, :]
        slices_updated.append(i)
    return slices_updated

def apply_transformation(slices, transform_functions):
    """
    The function applies the given transformation to the slices of the given set of patients. It is called in the train, val and test set data loaders in 2d images. 
    Args:
        slices (list): list of slices of the given set patients.
        transform_functions (function): transformation function to be applied to the slices.

    Returns:
        list: list of transformed slices of the given set patients.
    
    """
    updated_slices = add_dim_to_slices(slices) # add a dimension to the slices to perform the transformations
    list_transformed_slices = []
    for s in updated_slices:
        s = transform_functions(s)
        list_transformed_slices.append(s)
    return list_transformed_slices # return 2d and transformed slices


def get_slices_from_subset(dataset_subset, basal_fu = False, return_type=None): # only for the image slices not mask or label. 
    """ 3d to 2d. The function takes a datadataset_subset and returns a list of slices of the given set pateints.
    The slices are in 2d image format (512,512) and they are ansembled in a list.

    Args:
        dataset_subset (data dataset_subset): data dataset_subset of the set of patients.

    Returns:
        list: image slices and the corresponding labels of the given set of patients.
              or image masks and the corresponding labels of the given set of patients.
    """
    # Brings all the slices of the test set patients in to together in 2d image format (512,512)
    image_slices = []
    labels = [] # labels per each slice
    masks = [] # masks per each slice
    patient_ids = []
    patient_slice_numbers = []
    image_slices_fu = []
    label_fu = []
    masks_fu = []
    p_id = dataset_subset.patient_id
    count = 0
    skipped_count = 0
    skipped_patients = []

    for batch in dataset_subset:
        # Handle skipped samples (orthonormal issues)
        if batch is None:
            skipped_count += 1
            skipped_patients.append(p_id[count])
            count += 1
            continue
            
        image, mask, label = batch
        patient_id = p_id[count]
        
        # Skip if image or mask is None
        if image is None or mask is None:
            skipped_count += 1
            skipped_patients.append(patient_id)
            count += 1
            continue
            
        for idx in range(image.shape[0]): #image.size(1) is the number of slices in the image.
            slice = image[idx, :, :] # concatinating the slices of the same patient.
            mask_slice = mask[idx, :, :] # concatinating the masks of the same patient.
            
            # Apply resize/padding to ensure all slices are 512x512
            slice_resized = resize_or_pad_slice(slice, target_size=(512, 512))
            mask_resized = resize_or_pad_slice(mask_slice, target_size=(512, 512))
            
            patient_ids.append(patient_id)
            image_slices.append(slice_resized)
            labels.append(label)  # for the same patient's slices, the label is the same.
            masks.append(mask_resized) 
            patient_slice_numbers.append(idx) # for each slice of the patient, the slice number is saved.
        
        count += 1

    # Print skip report
    if skipped_count > 0:
        print(f"\n{'='*60}")
        print(f"ORTHONORMAL DIRECTION COSINES WARNING")
        print(f"{'='*60}")
        print(f"Total samples skipped: {skipped_count}")
        print(f"Skipped patient IDs: {skipped_patients}")
        print(f"Total samples processed: {len(labels)}")
        print(f"{'='*60}\n")

    # convert to tensor
    image_slices = torch.stack([torch.from_numpy(arr) for arr in image_slices])
    masks = np.array(masks, dtype=np.float32)
    masks = torch.stack([torch.from_numpy(arr) for arr in masks])
    # Convert patient_ids - keep as strings/objects if they're not numeric
    # Try to convert to int64, but if that fails, keep as object dtype
    try:
        patient_ids = np.array(patient_ids, dtype=np.int64)
        patient_ids = torch.from_numpy(patient_ids)
    except (ValueError, TypeError):
        # Patient IDs are strings, keep them as a list (PyTorch doesn't handle string tensors well)
        patient_ids = patient_ids  # Keep as list
    patient_slice_numbers = torch.from_numpy(np.array(patient_slice_numbers))
    labels = torch.from_numpy(np.array(labels).astype(np.int16))

    if return_type == 'image':
        return image_slices, labels, patient_ids, patient_slice_numbers
    elif return_type == 'mask':
        return masks, labels, patient_ids, patient_slice_numbers
    