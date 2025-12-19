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

    for batch in dataset_subset:
        image, mask, label = batch
        patient_id = p_id[count]
        for idx in range(image.shape[0]): #image.size(1) is the number of slices in the image.
            slice = image[idx, :, :] # concatinating the slices of the same patient.
            mask_slice = mask[idx, :, :] # concatinating the masks of the same patient.
            patient_ids.append(patient_id)
            image_slices.append(slice)
            labels.append(label)  # for the same patient's slices, the label is the same.
            masks.append(mask_slice) 
            patient_slice_numbers.append(idx) # for each slice of the patient, the slice number is saved.
        
        count += 1

    # convert to tensor
    image_slices = torch.stack([torch.from_numpy(arr) for arr in image_slices])
    masks = np.array(masks, dtype=np.float32)
    masks = torch.stack([torch.from_numpy(arr) for arr in masks])
    # Convert patient_ids to a supported type, such as int or float
    patient_ids = np.array(patient_ids, dtype=np.int64)
    # Convert the modified patient_ids and patient_slice_numbers to PyTorch Tensors
    patient_ids = torch.from_numpy(patient_ids)
    patient_slice_numbers = torch.from_numpy(np.array(patient_slice_numbers))
    labels = torch.from_numpy(np.array(labels).astype(np.int16))

    if return_type == 'image':
        return image_slices, labels, patient_ids, patient_slice_numbers
    elif return_type == 'mask':
        return masks, labels, patient_ids, patient_slice_numbers
    