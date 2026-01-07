import os
import sys; sys.path.insert(0, os.path.abspath("../"))
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from utils import *
import sys;
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as T
import random
import monai.transforms as transforms_monai
import cv2
import copy
import random
from torchvision.transforms import v2
import pytorch_lightning as pl

notebooks_path = Path.cwd()
repo_path = notebooks_path.parent
data_path = repo_path / 'data' / 'HematomaTruetaBasal' 
sys.path.insert(0, repo_path)

seed = 42 
pl.seed_everything(seed)

def find_normalization_parameters(image):
    """
    It takes an image and returns the mean and standard deviation of the image.
    
    :param image: the image to be normalized
    :return: The mean and standard deviation of the image. For 3d images.
    """

    norm_img = copy.deepcopy(image)
    # norm_img[norm_img == 0] = np.NaN
    # norm_parms = (np.nanmean(norm_img, axis=(-3, -2, -1), keepdims=True), 
    #               np.nanstd(norm_img, axis=(-3, -2, -1), keepdims=True))
    norm_parms = (np.nanmin(norm_img, axis=(-3, -2, -1), keepdims=True), 
                   np.nanmax(norm_img, axis=(-3, -2, -1), keepdims=True))

    return norm_parms 

def find_normalization_parameters2(image):
    """
    It takes an image and returns the mean and standard deviation of the image.
    
    :param image: the image to be normalized
    :return: The mean and standard deviation of the image. For 2d slices. 
    """

    norm_img = copy.deepcopy(image)
    norm_parms = (np.nanmin(norm_img, axis=(-2, -1), keepdims=True), 
                   np.nanmax(norm_img, axis=(-2, -1), keepdims=True))

    return norm_parms 

def normalize_image(image_patch, parameters):
    """
    The function takes an image patch and a list of parameters as input, and returns the normalized
    image patch.
    
    :param image_patch: the image patch that we want to normalize
    :param parameters: [mean, std]
    :return: The normalized image patch.
    """
    if len(image_patch.shape) == 3: # 2D case
        parameters = (np.squeeze(parameters[0], axis=-1),
                      np.squeeze(parameters[1], axis=-1))

    minmax = (image_patch-parameters[0]) / (parameters[1]-parameters[0]) # [0,1]
    minusone = (image_patch - (parameters[0]+parameters[1])/2)/((parameters[1]-parameters[0])/2) # [-1,1]

    # return (image_patch - parameters[0]) / parameters[1] 
    return minmax



class PredictionDataset(Dataset):
    """
    Dataset for prediction of the model.
    Args:
        md_path (Path): Path to the metadata file.
        mask (bool): If True, the dataset will return the mask as well.
        
        Returns:
            image (torch.tensor): Image of the patient.
            mask (torch.tensor): Mask of the patient.
            label (torch.tensor): Label of the patient.
    """
    def __init__(self, 
                 md_path: Path = None,  
                 fu_path: Path = None, 
                 basal_path: Path = None,
                 mask: bool = False, 
                 resize_image: bool = False, 
                 use_whole_image: bool = False):
        
        self.md_path = md_path 
        self.md_df = pd.read_csv(self.md_path) # reads the metadata
        self.fu_path = fu_path
        self.basal_path = basal_path
        self.labels = self.md_df['label'].values  # takes the labels from the metadata as an array.
        self.patient_id= self.md_df['patient_id'].values
        self.index = self.md_df['index'].values
        self.mask = mask
        self.resize_image = resize_image
        self.use_whole_image = use_whole_image # without filtering the slices that are not in the mask
        
        if self.fu_path is not None: # case: working with basal and follow-up images in small balances dataset. 
            self.fu_df = pd.read_csv(self.fu_path)
            self.basal_df = pd.read_csv(self.basal_path)

        if  self.mask == True:
            self.mask_paths = self.md_df['mask_path'].values
            self.image_paths = self.md_df['ct_ss_path'].values
        else:   
            self.image_paths = self.md_df['ct_ss_path'].values # takes the image paths from the metadata as an array.
        
    def __len__(self):
        return len(self.md_df)
    
    def __getitem__(self, idx):
            if self.mask == True:
                
                # load image
                image_path = self.image_paths[idx]
                # if image exists return the image if not pass to other image
                image = sitk.ReadImage(image_path)
                image.SetDirection(np.eye(3).flatten())  # CANSUreset to identity added later
                image = sitk.GetArrayFromImage(image)
                image = np.array(image)

                mask_path = self.mask_paths[idx]
                mask = sitk.ReadImage(mask_path)
                mask.SetDirection(np.eye(3).flatten()) #CANSU ADDED LATER
                mask = sitk.GetArrayFromImage(mask)
                mask = np.array(mask)
                
                # load label
                label = self.labels[idx]
                label = np.array(label)

                return image, mask, label

            else:
                # load image
                image_path = self.image_paths[idx]
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image)
                image = np.array(image)

                # load label
                label = self.labels[idx]
                label = np.array(label)
                return image, label

class DataAugmentation(object):
    """
    For each slice in the 3d image, it applies the transformations with 50% probability.
    """
    def __init__(self, apply_hflip, apply_affine, apply_gaussian_blur, degree, translate, scale, shear, hflip_p, affine_p ):
        self.apply_hflip = apply_hflip
        self.apply_affine = apply_affine
        self.apply_gaussian_blur = apply_gaussian_blur
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.percentage_hflip = hflip_p
        self.percentage_affine = affine_p

    def __call__(self, img, mask, image_index):
        if self.apply_hflip:
            if random.random() < self.percentage_hflip:
                img = T.hflip(img)
                mask = T.hflip(mask)

        if self.apply_gaussian_blur:
            if random.random() < 0.10:
                img = T.gaussian_blur(img, kernel_size=3, sigma=(0.1, 0.9))

        if self.apply_affine:
            if random.random() <  self.percentage_affine:
                img = T.affine(img, angle=self.degree, translate=self.translate, scale=self.scale, shear=self.shear)
                mask = T.affine(mask, angle=self.degree, translate=self.translate, scale=self.scale, shear=self.shear)
                
        return img, mask
    

class PredictionDataset2D(Dataset):
    """
    Dataset for trasnforming the 3d image into 2d slices and apllying the transformation.    
    """
    def __init__(self, 
                 slices, 
                 masks,
                 problem,
                 roi_size,
                 image_size,
                 patient_ids, 
                 slice_number,
                 test_type,
                 apply_hflip = False,
                 apply_affine = False,
                 apply_gaussian_blur = False,
                 affine_degree = 10,
                 affine_translate = 10,
                 affine_scale = 1.2,
                 affine_shear = 0,
                 basal_fu=False, 
                 labels=None, 
                 slices_fu=None,
                 masks_fu= None,
                 transform=False, 
                 three_channel_mask=False,
                 lesion=False,
                 roi=False,
                 hflip_p=0.5,
                 affine_p=0.5,
                 ):
        
        self.slices = slices
        self.test_type = test_type
        self.lesion = lesion
        self.basal_fu = basal_fu
        self.labels = labels
        self.masks = masks
        self.patient_ids = patient_ids
        self.slice_number = slice_number
        self.transform = transform
        self.three_channel_mask = three_channel_mask
        self.slices_fu = slices_fu
        self.masks_fu = masks_fu
        self.roi = roi
        self.roi_size = roi_size
        self.image_size = image_size
        self.problem = problem
        self.apply_hflip = apply_hflip
        self.apply_affine = apply_affine
        self.apply_gaussian_blur = apply_gaussian_blur
        self.affine_degree = affine_degree
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shear = affine_shear
        self.hflip_p = hflip_p
        self.affine_p = affine_p

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):

        id_value = self.patient_ids[idx].item()
        slice_number = self.slice_number[idx].item()
        combined_number = str(id_value) + str(slice_number)

    # self.basal_fu = false for the syhthetic dataset experiments. 
    # for basal_fu == false (prediction problem) -- only using the basal images dataset (the first setting we tried)
        updated_slice = self.slices[idx].unsqueeze(0)
        maskbasal = self.masks[idx].unsqueeze(0)

        if self.transform: # for training 2d prediction training 

            # no DA
            if self.test_type == 't1' or self.test_type == 't3': # Original data (no DA, no synthesis) or Original data + synthesis (no DA)
                image = updated_slice
                mask = maskbasal

            # DA    
            elif self.test_type == 't2' or self.test_type == 't5': # Original data + DA (no synthesis) or Original data + DA + synthesis
                image, mask = DataAugmentation(apply_hflip= self.apply_hflip, 
                                               apply_affine= self.apply_affine,
                                               apply_gaussian_blur= self.apply_gaussian_blur,
                                               degree = self.affine_degree,
                                                translate = (self.affine_translate, self.affine_translate),
                                                scale = self.affine_scale,
                                                shear = self.affine_shear, hflip_p=self.hflip_p, affine_p=self.affine_p)(img=updated_slice, mask=maskbasal, image_index= combined_number)
            else: 
                print("test type is not defined")
            
            # normalize each channel separately
            norm_params1 = find_normalization_parameters(image)
            image = normalize_image(image, norm_params1)

            mask2 = mask.cpu().numpy()
            mask2 = np.squeeze(mask2)
            mask2 = mask2.astype(np.uint8)
            mask2 = torch.from_numpy(mask2)

            changed_dim = image.expand(2,*image.shape[1:])
            mask2.unsqueeze_(0)
            three_channel = torch.cat((changed_dim, mask2), 0)
            return three_channel, self.labels[idx], self.patient_ids[idx], self.slice_number[idx]

        else: # for validation and test sets 
            # normalize each channel separately
            norm_params1 = find_normalization_parameters(updated_slice)
            updated_slice = normalize_image(updated_slice, norm_params1)

            mask = maskbasal.cpu().numpy()
            mask = np.squeeze(mask)
            mask = mask.astype(np.uint8)
            mask = torch.from_numpy(mask)
            mask.unsqueeze_(0)
            changed_dim = updated_slice.expand(2,*updated_slice.shape[1:])
            three_channel = torch.cat((changed_dim, mask), 0)
            
            return three_channel, self.labels[idx], self.patient_ids[idx], self.slice_number[idx]

class HEPredDataModule(pl.LightningDataModule):

    def __init__(self, 
                 split_indexes, 
                 batch_size,  
                 roi_size,
                 image_size,
                 test_type,
                 apply_hflip,
                 apply_affine, affine_degree, affine_translate , affine_scale, affine_shear,
                 apply_gaussian_blur,
                 hflip_p, affine_p,
                 problem: str = "prediction",
                 filter_slices: bool = False,
                 num_workers: int = 4, 
                 md_path: Path = None,
                 fu_path: Path = None, 
                 basal_path: Path = None,
                 mask: bool = False, 
                 use_2d: bool = True, 
                 return_type='image',
                 over_sampling: bool = False,
                 under_sampling: bool = False,
                 threshold: int = 2,
                 basal_fu: bool = False,
                 roi: bool = False, lesion: bool = False,

                 ):
        
        super().__init__()
        self.md_path = md_path
        self.problem = problem
        self.test_type = test_type
        self.roi = roi
        self.lesion = lesion
        self.roi_size = roi_size
        self.image_size = image_size
        self.fu_path = fu_path
        self.basal_path = basal_path
        self.batch_size = batch_size
        self.split_indexes = split_indexes
        self.num_workers = num_workers
        self.basal_fu = basal_fu
        self.mask = mask
        self.use_2d = use_2d # if True, the dataloader will return 2d slices of the 3d image 
        self.return_type = return_type # if return_type = 'image', the dataloader will return the image, if return_type = 'mask', the dataloader will return the mask
        self.filter_slices = filter_slices # if True, the dataloader will filter the slices that are segmented in the mask.
        self.over_sampling =  over_sampling # if True, the dataloader will use the  over_sampling to balance the dataset.
        self.under_sampling =  under_sampling # if True, the dataloader will use the  WeightedRandomSampler_under to balance the dataset.
        self.threshold = threshold # threshold used to filter the slices that are segmented in the mask. (# of pixels that are 1)
        self.apply_hflip = apply_hflip
        self.apply_affine = apply_affine
        self.affine_degree = affine_degree
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shear = affine_shear
        self.apply_gaussian_blur = apply_gaussian_blur
        self.hflip_p = hflip_p
        self.affine_p = affine_p

    def setup(self, stage=None):
        if self.basal_fu: # basal + fu cases
            self.dataset = PredictionDataset(md_path=self.md_path, fu_path=self.fu_path, basal_path=self.basal_path, mask=self.mask)
        else:
            self.dataset = PredictionDataset(md_path=self.md_path, mask=self.mask)

        # gets the indexes (actually patient_id) of the train, validation and test set
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_indexes

        # get the subsets of the train, validation and test setl
        self.train = torch.utils.data.Subset(self.dataset, indices=X_train)
        self.val = torch.utils.data.Subset(self.dataset, indices= X_val)
        self.test = torch.utils.data.Subset(self.dataset, indices= X_test)

        # Assigning the patient id and the labels to the train, validation and test set
        self.train.patient_id = self.dataset.patient_id[X_train] # same as y_train 
        self.val.patient_id = self.dataset.patient_id[X_val] # same as y_val
        self.test.patient_id = self.dataset.patient_id[X_test] # same as y_test

        self.train.labels = self.dataset.labels[X_train]
        self.val.labels = self.dataset.labels[X_val]
        self.test.labels = self.dataset.labels[X_test]

    def filter_segmented_slices(self, 
                                slices, 
                                mask, 
                                patient_ids, 
                                slice_numbers,
                                labels=None, 
                                slices_fu = None,
                                mask_fu = None,
                                threshold: int = 2):
        """
        This function filters the slices that are segmented in the mask. 
        Inputs: 
            slices: 3d tensor of the slices
            mask: 3d tensor of the mask
            patient_ids: 1d tensor of the patient ids
            labels: 1d tensor of the labels
            threshold: threshold used to filter the slices that are segmented in the mask. (# of pixels that are 1)
        Outputs:
            filtered_slices: 3d tensor of the slices that are segmented in the mask
            filtered_masks: 3d tensor of the mask that are segmented in the mask
            filtered_ids: 1d tensor of the patient ids that are segmented in the mask
            filtered_labels: 1d tensor of the labels that are segmented in the mask
        """
        
        if self.basal_fu:
            # Find the slices where the lesion is available
            lesion_slices = []
            for i in range(len(slices)):
                if (mask[i].sum() > threshold) and (mask_fu[i].sum() > threshold): # take if there are at least 2 pixels with value 1 
                    lesion_slices.append(i)

             # Select only the slices where the lesion is available
            filtered_slices = slices[lesion_slices]
            filtered_masks = mask[lesion_slices]
            filtered_ids = patient_ids[lesion_slices]
            filtered_labels = labels[lesion_slices]
            filtered_slice_numbers = slice_numbers[lesion_slices]
            filtered_slices_fu = slices_fu[lesion_slices]
            filtered_masks_fu = mask_fu[lesion_slices]

            # convert the slices to tensor
            filtered_slices= filtered_slices.clone().detach()
            filtered_masks= filtered_masks.clone().detach()
            filtered_ids= filtered_ids.clone().detach()
            filtered_slice_numbers= filtered_slice_numbers.clone().detach()
            filtered_slices_fu= filtered_slices_fu.clone().detach()
            filtered_masks_fu= filtered_masks_fu.clone().detach()
            filtered_labels= filtered_labels.clone().detach()

            return filtered_slices, filtered_masks, filtered_ids, filtered_slice_numbers, filtered_slices_fu, filtered_masks_fu, filtered_labels
        
        else:
            # Find the slices where the lesion is available
            lesion_slices = []
            for i in range(len(slices)):
                if mask[i].sum() > threshold: # take if there are at least 2 pixels with value 1 
                    lesion_slices.append(i)

            # Select only the slices where the lesion is available
            filtered_slices = slices[lesion_slices]
            filtered_masks = mask[lesion_slices]
            filtered_labels = labels[lesion_slices]
            filtered_ids = patient_ids[lesion_slices]
            filtered_slice_numbers = slice_numbers[lesion_slices]

            # convert the slices to tensor
            filtered_slices= filtered_slices.clone().detach()
            filtered_masks= filtered_masks.clone().detach()
            filtered_labels= filtered_labels.clone().detach()
            filtered_ids= filtered_ids.clone().detach()
            filtered_slice_numbers= filtered_slice_numbers.clone().detach()
            
            return filtered_slices, filtered_masks, filtered_labels, filtered_ids, filtered_slice_numbers

    def train_dataloader(self):

        seed = 42
        torch.manual_seed(seed)
        seed_everything(seed)
    
        if self.use_2d==True:
            if self.filter_slices==True: # Return only the slices where the lesion is available
                slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.train, basal_fu=self.basal_fu, return_type=self.return_type) # returns all the slices in the given set.
                mask_slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.train, return_type='mask', basal_fu=self.basal_fu)
                filtered_slices, filtered_masks, filtered_labels, filtered_ids, filtered_slice_numbers = self.filter_segmented_slices(slices=slices, 
                                                                                                            mask=mask_slices, 
                                                                                                            patient_ids=patient_ids,
                                                                                                            labels=labels,
                                                                                                            slice_numbers=patient_slice_numbers,
                                                                                                            threshold=self.threshold)

                self.dataset_2d  = PredictionDataset2D(slices=filtered_slices, 
                masks=filtered_masks,
                patient_ids=filtered_ids, 
                labels=filtered_labels,
                slice_number=filtered_slice_numbers,
                basal_fu=self.basal_fu,
                transform=True, roi=self.roi, 
                problem=self.problem, 
                roi_size=self.roi_size, 
                image_size=self.image_size, 
                lesion=self.lesion,
                test_type = self.test_type, 
                apply_hflip=self.apply_hflip,
                apply_affine=self.apply_affine,
                apply_gaussian_blur=self.apply_gaussian_blur,
                affine_degree=self.affine_degree,
                affine_translate=self.affine_translate,
                affine_scale=self.affine_scale,
                affine_shear=self.affine_shear, 
                hflip_p=self.hflip_p, affine_p = self.affine_p)
            
                print( "number of slices in the train set: ", len(self.dataset_2d))
                return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True,drop_last=True)
            
            else: # Return all the slices (the whole image volume)
                self.dataset_2d = PredictionDataset2D(slices=filtered_slices, labels=filtered_labels, patient_ids=filtered_ids, transform=True)
                print( "number of slices in the train set: ", len(self.dataset_2d))
                return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)
            
        else: # return 3d volumes
                if self.basal_fu==True:
                    return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, 
                            shuffle=True, drop_last=True)
                else:
                    return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, 
                            shuffle=True) # return 3d volumes

    def val_dataloader(self):
        if self.use_2d==True:
            slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.val, basal_fu=self.basal_fu, return_type=self.return_type) # returns all the slices in the given set.
            if self.filter_slices==True: # Return only the slices where the lesion is available
                mask_slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.val, return_type='mask', basal_fu=self.basal_fu)
                filtered_slices, filtered_masks, filtered_labels, filtered_ids, filtered_slice_numbers = self.filter_segmented_slices(slices=slices, 
                                                                                                            mask=mask_slices, 
                                                                                                            patient_ids=patient_ids,
                                                                                                            labels=labels,
                                                                                                            slice_numbers=patient_slice_numbers,
                                                                                                            threshold=self.threshold)
                print( "number of slices after filtering in validation ", len(filtered_labels))
                self.dataset_2d  = PredictionDataset2D(slices=filtered_slices, 
                                      masks=filtered_masks,
                                      patient_ids=filtered_ids, 
                                      labels=filtered_labels,
                                      slice_number=filtered_slice_numbers,
                                      basal_fu=self.basal_fu,
                                      transform=False, 
                                      roi=self.roi, 
                                      problem=self.problem, 
                                      roi_size=self.roi_size, 
                                      image_size=self.image_size, 
                                      lesion=self.lesion,
                                      test_type = self.test_type)
                
                print( "number of slices in the validation set: ", len(self.dataset_2d))
                return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False,drop_last=True)
            
            else: # Return all the slices
                self.dataset_2d = PredictionDataset2D(slices, labels, transform=False)
                return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)
        else: # Return 3d volumes
            if self.basal_fu==True:
                return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, 
                        shuffle=False)
            else:
                return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, 
                        shuffle=False) # return 3d volumes
    
    def test_dataloader(self):
        if self.use_2d==True:
            slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.test, basal_fu=self.basal_fu, return_type=self.return_type) # returns all the slices in the given set.
            if self.filter_slices==True: # Return only the slices where the lesion is available
                mask_slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.test, return_type='mask', basal_fu=self.basal_fu)
                filtered_slices, filtered_masks, filtered_labels, filtered_ids, filtered_slice_numbers = self.filter_segmented_slices(slices=slices, 
                                                                                                            mask=mask_slices, 
                                                                                                            patient_ids=patient_ids,
                                                                                                            labels=labels,
                                                                                                            slice_numbers=patient_slice_numbers,
                                                                                                            threshold=self.threshold)
                print( "number of slices after filtering in test ", len(filtered_labels))
                self.dataset_2d  = PredictionDataset2D(slices=filtered_slices, 
                                      masks=filtered_masks,
                                      patient_ids=filtered_ids, 
                                      labels=filtered_labels,
                                      slice_number=filtered_slice_numbers,
                                      basal_fu=self.basal_fu,
                                      transform=False, 
                                      roi=self.roi, 
                                      problem=self.problem, 
                                      roi_size=self.roi_size, 
                                      image_size=self.image_size, 
                                      lesion=self.lesion,
                                      test_type = self.test_type)
                
                print( "number of slices in the test set: ", len(self.dataset_2d))
                return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)
            
            else: # Return all the slices
                self.dataset_2d = PredictionDataset2D(slices, labels, transform=False)
                return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)
        else: # Return 3d volumes
            if self.basal_fu==True:
                return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, 
                        shuffle=False)
            else:
                return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, 
                        shuffle=False) # return 3d volumes
            
            