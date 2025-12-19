import logging
import torch
import cv2
import os
import numpy as np
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy
from torchvision import models, transforms
import pandas as pd
import pytorch_lightning as pl
import json
from monai.networks.nets import SEResNet50
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torchmetrics
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import timm
from torch.autograd import Function
from monai.networks.blocks import SEBlock
from torch.optim.lr_scheduler import ReduceLROnPlateau
repo_path = os.getcwd()

class SELayer(nn.Module):
        def __init__(self, channel, reduction=16):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

def focal_loss(logits, targets, alpha=0.7, gamma=1, device = 0):
    # bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float().to(f"cuda:{device}"), reduction='none')
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return torch.mean(focal_loss)

def compute_sensitivity_specificity(preds, labels):
    true_positives = (preds == 1) & (labels == 1)
    true_negatives = (preds == 0) & (labels == 0)
    false_positives = (preds == 1) & (labels == 0)
    false_negatives = (preds == 0) & (labels == 1)
    sensitivity = true_positives.sum() / (true_positives.sum() + false_negatives.sum())
    specificity = true_negatives.sum() / (true_negatives.sum() + false_positives.sum())
    return sensitivity, specificity

"""
Commented the models that are not used in this study.
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(UNetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pool4(x)

        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)

        return x

class UNetEncoder_paper(nn.Module):
    # Modified Unet encoder module used in the study "Deep learning for automatically predicting early haematoma expansion in Chinese patients"
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetEncoder_paper, self).__init__()

        # # # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # # # self.instance_norm1 = nn.InstanceNorm2d(64)
        # # # self.batch_norm1 = nn.BatchNorm2d(64)
        # # # self.leaky_relu_1 = nn.LeakyReLU(inplace=True)
        # # # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # # # self.instance_norm2 = nn.InstanceNorm2d(64)
        # # # self.batch_norm2 = nn.BatchNorm2d(64)
        # # # self.leaky_relu_2 = nn.LeakyReLU(inplace=True)
        # # # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # # # self.use_conv_1x1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        # # # #add residual connection

        # # # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # # # self.instance_norm3 = nn.InstanceNorm2d(128)
        # # # self.batch_norm3 = nn.BatchNorm2d(128)
        # # # self.leaky_relu_3 = nn.LeakyReLU(inplace=True)
        # # # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # # # self.instance_norm4 = nn.InstanceNorm2d(128)
        # # # self.batch_norm4 = nn.BatchNorm2d(128)
        # # # self.leaky_relu_4 = nn.LeakyReLU(inplace=True)
        # # # self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # # # self.instance_norm5 = nn.InstanceNorm2d(128)
        # # # self.batch_norm5 = nn.BatchNorm2d(128)
        # # # self.use_conv_1x1_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        # # # self.leaky_relu_5 = nn.LeakyReLU(inplace=True)
       
        # # # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # # # self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # # # self.instance_norm6 = nn.InstanceNorm2d(256)
        # # # self.batch_norm6 = nn.BatchNorm2d(256)
        # # # self.leaky_relu_6 = nn.LeakyReLU(inplace=True)
        # # # self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # # # self.instance_norm7 = nn.InstanceNorm2d(256)
        # # # self.batch_norm7 = nn.BatchNorm2d(256)
        # # # self.leaky_relu_7 = nn.LeakyReLU(inplace=True)
        # # # self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # # # self.instance_norm8 = nn.InstanceNorm2d(256)
        # # # self.batch_norm8 = nn.BatchNorm2d(256)
        # # # # add residual connection
        # # # self.leaky_relu_8 = nn.LeakyReLU(inplace=True)

        # # # self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # # # self.instance_norm9 = nn.InstanceNorm2d(512)
        # # # self.batch_norm9 = nn.BatchNorm2d(512)
        # # # self.leaky_relu_9 = nn.LeakyReLU(inplace=True)

        # # # self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # # # self.instance_norm10 = nn.InstanceNorm2d(512)
        # # # self.batch_norm10 = nn.BatchNorm2d(512)
        # # # self.leaky_relu_10 = nn.LeakyReLU(inplace=True)
        
        # # # self.se_block = SELayer(512,reduction=16)

        # # # self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # # # self.instance_norm11 = nn.InstanceNorm2d(512)
        # # # self.batch_norm11 = nn.BatchNorm2d(512)
        # # # self.leaky_relu_11 = nn.LeakyReLU(inplace=True)

        # # # self.average_pool = nn.AdaptiveAvgPool2d(1)
        # # # self.fc = nn.Linear(512, out_channels)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(64)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu_1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.leaky_relu_2 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_conv_1x1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        #add residual connection

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.instance_norm3 = nn.InstanceNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.leaky_relu_3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm4 = nn.InstanceNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.leaky_relu_4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm5 = nn.InstanceNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.use_conv_1x1_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.leaky_relu_5 = nn.LeakyReLU(inplace=True)
       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.instance_norm6 = nn.InstanceNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.leaky_relu_6 = nn.LeakyReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm7 = nn.InstanceNorm2d(256)
        self.batch_norm7 = nn.BatchNorm2d(256)
        self.leaky_relu_7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm8 = nn.InstanceNorm2d(256)
        self.batch_norm8 = nn.BatchNorm2d(256)
        # add residual connection
        self.leaky_relu_8 = nn.LeakyReLU(inplace=True)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.instance_norm9 = nn.InstanceNorm2d(512)
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.leaky_relu_9 = nn.LeakyReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm10 = nn.InstanceNorm2d(512)
        self.batch_norm10 = nn.BatchNorm2d(512)
        self.leaky_relu_10 = nn.LeakyReLU(inplace=True)
        
        self.se_block = SELayer(512,reduction=16)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm11 = nn.InstanceNorm2d(512)
        self.batch_norm11 = nn.BatchNorm2d(512)
        self.leaky_relu_11 = nn.LeakyReLU(inplace=True)

        # self.conv12 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        # self.instance_norm12 = nn.InstanceNorm2d(1024)
        # self.batch_norm12 = nn.BatchNorm2d(1024)
        # self.leaky_relu_12 = nn.LeakyReLU(inplace=True)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_channels)

    def forward(self, x):
            
            input = torch.squeeze(x, -1)
            out = self.conv1(input)
            out = self.instance_norm1(out)
            out = self.leaky_relu_1(out)
            out = self.conv2(out)
            out = self.instance_norm2(out)
            out = self.leaky_relu_2(out)
            out = self.pool1(out)

            # apply conv 1x1 
            residual1 = self.use_conv_1x1(out)

            out = self.conv3(out)
            out = self.instance_norm3(out)
            out = self.leaky_relu_3(out)
            out = self.conv4(out)
            out = self.instance_norm4(out)
            out = self.leaky_relu_4(out)
            out = self.conv5(out)
            out = self.instance_norm5(out)
            out = out + residual1
            out = self.leaky_relu_5(out)
            
            out = self.pool2(out)
            # apply conv 1x1
            residual2 = self.use_conv_1x1_2(out)

            out = self.conv6(out)
            out = self.instance_norm6(out)
            out = self.leaky_relu_6(out)
            out = self.conv7(out)
            out = self.instance_norm7(out)
            out = self.leaky_relu_7(out)
            out = self.conv8(out)
            out = self.instance_norm8(out)
            out = out + residual2
            out = self.leaky_relu_8(out)
            
            out = self.conv9(out)
            out = self.instance_norm9(out)
            out = self.leaky_relu_9(out)

            out = self.conv10(out)
            out = self.instance_norm10(out)
            out = self.leaky_relu_10(out)

            # SE layer is added. 
            # out = self.se_block(out)

            out = self.conv11(out)
            out = self.instance_norm11(out)
            out = self.leaky_relu_11(out)

            # out = self.conv12(out)
            # out = self.instance_norm12(out)
            # out = self.leaky_relu_12(out)

            out = self.average_pool(out)
            out = out.view(out.size(0),-1)
            out = self.fc(out)
            out = out.squeeze()
            return out

class UNetEncoder_paper2(nn.Module):
    # Modified Unet encoder module used in the study "Deep learning for automatically predicting early haematoma expansion in Chinese patients"
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetEncoder_paper2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(64)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu_1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.leaky_relu_2 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_conv_1x1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        #add residual connection

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.instance_norm3 = nn.InstanceNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.leaky_relu_3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm4 = nn.InstanceNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.leaky_relu_4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm5 = nn.InstanceNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.use_conv_1x1_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.leaky_relu_5 = nn.LeakyReLU(inplace=True)
       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.instance_norm6 = nn.InstanceNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.leaky_relu_6 = nn.LeakyReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm7 = nn.InstanceNorm2d(256)
        self.batch_norm7 = nn.BatchNorm2d(256)
        self.leaky_relu_7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm8 = nn.InstanceNorm2d(256)
        self.batch_norm8 = nn.BatchNorm2d(256)
        # add residual connection
        self.leaky_relu_8 = nn.LeakyReLU(inplace=True)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.instance_norm9 = nn.InstanceNorm2d(512)
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.leaky_relu_9 = nn.LeakyReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm10 = nn.InstanceNorm2d(512)
        self.batch_norm10 = nn.BatchNorm2d(512)
        self.leaky_relu_10 = nn.LeakyReLU(inplace=True)
        
        self.se_block = SELayer(512,reduction=16)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm11 = nn.InstanceNorm2d(512)
        self.batch_norm11 = nn.BatchNorm2d(512)
        self.leaky_relu_11 = nn.LeakyReLU(inplace=True)

        # self.conv12 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        # self.instance_norm12 = nn.InstanceNorm2d(1024)
        # self.batch_norm12 = nn.BatchNorm2d(1024)
        # self.leaky_relu_12 = nn.LeakyReLU(inplace=True)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_channels)

    def forward(self, x):
            
            input = torch.squeeze(x, -1)
            out = self.conv1(input)
            out = self.instance_norm1(out)
            out = self.leaky_relu_1(out)
            out = self.conv2(out)
            out = self.instance_norm2(out)
            out = self.leaky_relu_2(out)
            out = self.pool1(out)

            # apply conv 1x1 
            residual1 = self.use_conv_1x1(out)

            out = self.conv3(out)
            out = self.instance_norm3(out)
            out = self.leaky_relu_3(out)
            out = self.conv4(out)
            out = self.instance_norm4(out)
            out = self.leaky_relu_4(out)
            out = self.conv5(out)
            out = self.instance_norm5(out)
            out = out + residual1
            out = self.leaky_relu_5(out)
            
            out = self.pool2(out)
            # apply conv 1x1
            residual2 = self.use_conv_1x1_2(out)

            out = self.conv6(out)
            out = self.instance_norm6(out)
            out = self.leaky_relu_6(out)
            out = self.conv7(out)
            out = self.instance_norm7(out)
            out = self.leaky_relu_7(out)
            out = self.conv8(out)
            out = self.instance_norm8(out)
            out = out + residual2
            out = self.leaky_relu_8(out)
            
            out = self.conv9(out)
            out = self.instance_norm9(out)
            out = self.leaky_relu_9(out)

            out = self.conv10(out)
            out = self.instance_norm10(out)
            out = self.leaky_relu_10(out)

            # SE layer is added. 
            out = self.se_block(out)

            out = self.conv11(out)
            out = self.instance_norm11(out)
            out = self.leaky_relu_11(out)

            # out = self.conv12(out)
            # out = self.instance_norm12(out)
            # out = self.leaky_relu_12(out)

            out = self.average_pool(out)
            out = out.view(out.size(0),-1)
            out = self.fc(out)
            out = out.squeeze()
            return out

class UNetEncoder_paper_updated(nn.Module):
    # Modified Unet encoder module used in the study "Deep learning for automatically predicting early haematoma expansion in Chinese patients"
    # added mlp layer as classifier
    def __init__(self, in_channels=3):
        super(UNetEncoder_paper_updated, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(64)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu_1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.leaky_relu_2 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_conv_1x1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        #add residual connection

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.instance_norm3 = nn.InstanceNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.leaky_relu_3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm4 = nn.InstanceNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.leaky_relu_4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm5 = nn.InstanceNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.use_conv_1x1_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.leaky_relu_5 = nn.LeakyReLU(inplace=True)
       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.instance_norm6 = nn.InstanceNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.leaky_relu_6 = nn.LeakyReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm7 = nn.InstanceNorm2d(256)
        self.batch_norm7 = nn.BatchNorm2d(256)
        self.leaky_relu_7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm8 = nn.InstanceNorm2d(256)
        self.batch_norm8 = nn.BatchNorm2d(256)
        # add residual connection
        self.leaky_relu_8 = nn.LeakyReLU(inplace=True)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.instance_norm9 = nn.InstanceNorm2d(512)
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.leaky_relu_9 = nn.LeakyReLU(inplace=True)

        # classifier part
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm10 = nn.InstanceNorm2d(512)
        self.batch_norm10 = nn.BatchNorm2d(512)
        self.leaky_relu_10 = nn.LeakyReLU(inplace=True)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm11 = nn.InstanceNorm2d(512)
        self.batch_norm11 = nn.BatchNorm2d(512)
        self.leaky_relu_11 = nn.LeakyReLU(inplace=True)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)
        self.mlp = torchvision.ops.MLP(in_channels=512, hidden_channels = [512, 1], activation_layer=torch.nn.LeakyReLU, dropout=0.0)

    def forward(self, x):

        outputs = []
        for b in range(x.shape[0]): # Go over the number of batch of the input

            input = x[b]

            out = self.conv1(input)
            out = self.instance_norm1(out)
            out = self.leaky_relu_1(out)
            out = self.conv2(out)
            out = self.instance_norm2(out)
            out = self.leaky_relu_2(out)
            out = self.pool1(out)

            # apply conv 1x1 
            residual1 = self.use_conv_1x1(out)

            out = self.conv3(out)
            out = self.instance_norm3(out)
            out = self.leaky_relu_3(out)
            out = self.conv4(out)
            out = self.instance_norm4(out)
            out = self.leaky_relu_4(out)
            out = self.conv5(out)
            out = self.instance_norm5(out)
            out = out + residual1
            out = self.leaky_relu_5(out)
            
            out = self.pool2(out)
            # apply conv 1x1
            residual2 = self.use_conv_1x1_2(out)

            out = self.conv6(out)
            out = self.instance_norm6(out)
            out = self.leaky_relu_6(out)
            out = self.conv7(out)
            out = self.instance_norm7(out)
            out = self.leaky_relu_7(out)
            out = self.conv8(out)
            out = self.instance_norm8(out)
            out = out + residual2
            out = self.leaky_relu_8(out)
            
            out = self.conv9(out)
            out = self.instance_norm9(out)
            out = self.leaky_relu_9(out)

            out = self.conv10(out)
            out = self.instance_norm10(out)
            out = self.leaky_relu_10(out)

            out = self.conv11(out)
            out = self.instance_norm11(out)
            out = self.leaky_relu_11(out)

            out = self.average_pool(out)
            out = out.view(-1, out.size(0))
            out = self.mlp(out)
            outputs.append(out)
        stacked_outputs = torch.stack(outputs, dim=0).squeeze(1).squeeze(1)
        return stacked_outputs

class ImageClassifier(pl.LightningModule):
    def __init__(self, model='resnet50', 
                 num_classes=2, 
                 learning_rate=1e-5,
                 weight_decay = 1e-3,
                 optimizer='adam',
                 class_weights=[1,1],
                 pw_based=None,
                 test_pw_based =None,
                 threshold=0.5,
                 rad_img_net=False,
                 hematoma_weights=False,
                 num_features=0, loss="focal"):
        
        super().__init__()
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.pw_based = pw_based
        self.test_pw_based = test_pw_based
        self.class_weights = class_weights
        self.threshold = threshold
        self.rad_img_net = rad_img_net
        self.hematoma_weights = hematoma_weights
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.best_roc_auc = 0
        self.num_features = num_features

        # # train eveything from scratch - Resnet

        if self.model == 'resnet18':
            backbone = torchvision.models.resnet18(weights="DEFAULT")
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        if self.model=='resnet50':
            print('Using resnet50')
            backbone = torchvision.models.resnet50(weights="DEFAULT")
            num_features = backbone.fc.in_features
            for name, param in backbone.named_parameters():
                if 'layer4.1' not in name: # layer4: 8m 'layer4.1':4m
                    param.requires_grad = False
            self.num_features = num_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone
        
        if self.model=='squeezenet':
            squeezenet = torchvision.models.squeezenet1_0(weights=None)
            squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
            self.backbone = squeezenet

        if self.model=='resnet50_s':
            print('Using resnet50 from scratch')
            backbone = torchvision.models.resnet50(weights=None)
            num_features = backbone.fc.in_features
            self.num_features = num_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        # # swin_t transformer # 15m params
        if self.model=='swin':
            print('Using swin_t')
            model = torchvision.models.swin_t(weights="DEFAULT")
            num_features = model.head.in_features
            for name, param in model.named_parameters():
                if 'features' in name:
                    layer_num = int(name.split('.')[1])  # Get the layer number from the name
                    if layer_num < 6:
                        param.requires_grad = False
            model.head = nn.Linear(num_features, 1)
            self.backbone = model

        if self.model=='swin_s':
            print('Using swin_t from scratch')
            model = torchvision.models.swin_t(weights="DEFAULT")
            num_features = model.head.in_features
            model.head = nn.Linear(num_features, 1)
            self.backbone = model

        # effiecienet b
        if self.model=='eff':
            print('Using eff')
            eff = torchvision.models.efficientnet_b0(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            for name, param in eff.named_parameters():
                if 'features.7' not in name: # features.6: 2M params features.7: 700k 
                    param.requires_grad = False
            self.backbone = eff

        if self.model=='eff_s':
            print('Using eff s')
            eff = torchvision.models.efficientnet_b0(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            self.backbone = eff

        if self.model == "eva":
            print("Using eva")
            eva_model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=True, num_classes=1)
            self.backbone = eva_model

        if self.model == "eva_s":
            print("Using eva_s")
            eva_model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=False, num_classes=1)
            self.backbone = eva_model

        if self.model == "r50-vit": # they used this hybrid transformer in Transunet paper (99M parameters)
            print("Using r50-vit")
            r50_vit = timm.create_model('vit_base_r50_s16_384.orig_in21k_ft_in1k', pretrained=True, num_classes=1)
            for name, param in r50_vit.named_parameters(): # just freeze the resnet50 backbone weights but train the vit 
                param.requires_grad = False
            for name, param in r50_vit.blocks[11].named_parameters(): #r50_vit_model.patch_embed # just last layer is open r50_vit_model.blocks[11]
                param.requires_grad = True
            self.backbone = r50_vit

        if self.model == "r50-vit_s": # they used this hybrid transformer in Transunet paper (99M parameters)
            print("Using r50-vit_s")
            r50_vit = timm.create_model('vit_base_r50_s16_384.orig_in21k_ft_in1k', pretrained=True, num_classes=1)
            self.backbone = r50_vit

        if self.model == "nest":
            print("Using nest")
            nest = nest = timm.create_model('timm/nest_tiny_jx.goog_in1k', pretrained=True, num_classes=1)
            self.backbone = nest

        if self.model == "unet":
            print("Using unet")
            self.backbone = UNetEncoder_paper(in_channels=3, out_channels=1)

        if self.model =="ag_unet":
            print("Using ag_unet")
            self.backbone = AG_UNET(in_channels=3, out_channels=1)

        if self.model == "densenet121":
            print("Using densenet121")
            densenet = torchvision.models.densenet121(pretrained=True)
            num_features = densenet.classifier.in_features
            densenet.classifier = nn.Linear(num_features, 1)
            self.backbone = densenet

        # Assign different metrics for training and validation sets to not mix them up
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')

        self.train_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.val_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.test_accuracy_pw = torchmetrics.Accuracy(task='binary')

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

        self.train_f1_pw = torchmetrics.F1Score(task='binary')
        self.val_f1_pw = torchmetrics.F1Score(task='binary')
        self.test_f1_pw = torchmetrics.F1Score(task='binary')

        self.train_precision_calculate = torchmetrics.Precision(task='binary')
        self.val_precision_calculate = torchmetrics.Precision(task='binary')
        self.test_precision_calculate = torchmetrics.Precision(task='binary')

        self.train_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.val_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.test_precision_calculate_pw = torchmetrics.Precision(task='binary')

        if self.loss=="focal":
            print("Using focal loss")
        else:
            print("Using cross entropy loss")
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x)

        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)
        
        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('train_images', grid, self.current_epoch) 

        acc = self.train_accuracy(preds, y)
        prec = self.train_precision_calculate(preds, y)
        f1 = self.train_f1(preds, y)
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}

    def validation_step(self, batch, batch_idx):

        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x) # logits: predicted probablities

        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)
            
        acc = self.val_accuracy(preds, y)
        prec = self.val_precision_calculate(preds, y)
        f1 = self.val_f1(preds, y)

        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('val_images', grid, self.current_epoch)
        
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}
    
    def test_step(self, batch, batch_idx):

        x, y, patient_id, patient_slice_nums = batch
        logits = self(x)
        
        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)

        acc = self.test_accuracy(preds, y)
        prec = self.test_precision_calculate(preds, y)
        f1 = self.test_f1(preds, y)

        self.log('test_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 
                'probs': probs, 
                'labels': y, 
                'patient_id': patient_id,
                'patient_slice_nums': patient_slice_nums}

    def training_epoch_end(self, outputs):
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)

            # Just take the second elements of the probabilities and create a new tensor with them. 
            # Second element is the probability of being positive # only one probality
            concat_single_probs = concat_probs

            # create a dictionary with patient ids as keys and probabilities as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            # create a dictionary with patient ids as keys and labels as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            # create a dictionary with patient ids as keys and slice numbers as values
            for i in range(len(concat_patient_slice_nums)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            # k: patient_id, v: probability and self.patient_labels[patient_id]: label
            # self.patient_probs = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # combine the probabilities and labels and slice numbers
            # k: patient_id, v: probability and self.patient_labels[patient_id]: label
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # Calculate the epoch metrics for patient level
            acc_pw = self.train_accuracy_pw(y_pred, y_true)
            acc_pw= self.train_accuracy_pw.compute()
            prec_pw = self.train_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.train_precision_calculate_pw.compute()
            f1_pw = self.train_f1_pw(y_pred, y_true)
            f1_pw = self.train_f1_pw.compute()

            self.log("train_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.train_accuracy.compute()
        prec = self.train_precision_calculate.compute()
        f1 = self.train_f1.compute()
        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_slice_nums = torch.cat(patient_slice_nums, dim=0)

            #just take the second elements of the probabilities and create a new tensor with them
            # only one probality
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                slice_num = concat_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities, labels and slice numbers
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            self.log('val_sens_iw', sens_iw, prog_bar=True, sync_dist=True) # validation sensitivity score
            self.log('val_spec_iw', spec_iw, prog_bar=True, sync_dist=True) # validation specificity score
            self.log('val_roc_auc_iw', roc_auc_iw, prog_bar=True, sync_dist=True) # validation auroc score

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            # Validation AUROC scores
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            self.log('val_roc_auc', roc_auc, prog_bar=True, sync_dist=True) # validation auroc score

            # Calculate the epoch metrics for patient level
            acc_pw = self.val_accuracy_pw(y_pred, y_true)
            acc_pw= self.val_accuracy_pw.compute()
            prec_pw = self.val_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.val_precision_calculate_pw.compute()
            f1_pw = self.val_f1_pw(y_pred, y_true)
            f1_pw = self.val_f1_pw.compute()
            
            self.log("val_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_sens_pw", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_spec_pw", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs] 
            stacked = torch.stack(losses)
            # takes the average of the losses in one epoch
            avg_loss = torch.mean(stacked)
            self.log('val_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.val_accuracy.compute()
        prec = self.val_precision_calculate.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary
        self.patient_probs_with_slices = {} # reset patient probabilities with slice numbers dictionary
    
        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_id'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]

            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_patient_slice_nums = concat_patient_slice_nums + 1 # add 1 to the slice numbers to make them start from 1 instead of 0

            #just take the second elements of the probabilities and create a new tensor with them
            # # only one probality
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # create a nested dictionary with patient id as the first key and slice number as the second key and probability as the value
            self.patient_probs_with_slices = {k: {k2: v2 for k2, v2 in zip(self.patient_slice_nums[k], v)} for k, v in self.patient_probs.items()}
            # sort the nested dictionary by slice number
            self.patient_probs_with_slices = {k: dict(sorted(v.items())) for k, v in self.patient_probs_with_slices.items()} 
            print("self.patient_probs_with_slices after sorting: ", self.patient_probs_with_slices)
            #save the nested dictionary as a txt file
            with open('patient_probs_with_slices.txt', 'w') as file:
                file.write(json.dumps(self.patient_probs_with_slices))

            # for patient_id in self.patient_probs_with_slices:
            #     fig = plt.figure()
            #     plt.scatter(list(self.patient_slice_nums[patient_id]), list(self.patient_probs[patient_id]))
            #     plt.xlabel("Slice Number")
            #     plt.ylabel("Probability")
            #     plt.title("Patient " + str(patient_id))
            #     plt.savefig(f"plots/{patient_id}.png")
            #     plt.close(fig)

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # fig = plt.figure()
            # plt.scatter(list(concat_labels[concat_labels == 0]), list(concat_single_probs[concat_labels == 0]))
            # plt.scatter(list(concat_labels[concat_labels == 1]), list(concat_single_probs[concat_labels == 1]))
            # plt.xlabel("Class Label")
            # plt.ylabel("Probability")
            # plt.title("Patients with Class Label 0 and 1")
            # plt.savefig(f"plots/class_label_0_1.png")
            # plt.close(fig)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.test_pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.test_pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.test_pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for test_pw_based")
                
            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            self.patient_probs_dict = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            #print assigned means for each patient
            # print("Assigned means for each patient: ", self.patient_probs)

            # Calculate the epoch metrics for patient level
            acc_pw = self.test_accuracy_pw(y_pred, y_true)
            acc_pw= self.test_accuracy_pw.compute()
            prec_pw = self.test_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.test_precision_calculate_pw.compute()
            f1_pw = self.test_f1_pw(y_pred, y_true)
            f1_pw = self.test_f1_pw.compute()

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            print(" TEST EPOCH END Predictions PW : ", y_pred)
            print(" TEST EPOCH END Labels PW: ", y_true)

            self.log("test_prediction_pw", y_pred, on_step=False, on_epoch=True, prog_bar=False)
            self.log("test_label_pw", y_true, on_step=False, on_epoch=True, prog_bar=False)

            # compute roc curve 
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            # ROC curve of test data
            score = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) # AUC score. Worst case scneario is 0.5, best case scenario is 1.0
            # plt.title('Receiver Operating Characteristic')
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.legend(loc='lower right')
            # # save the plot
            # plt.savefig(f"plots/roc_curve.png")

            # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

            self.log('test_roc_auc_score_pw', score, prog_bar=True, sync_dist=True)
            self.log('test_roc_auc_score_iw', roc_auc_iw, prog_bar=True, sync_dist=True)
            self.log("test_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_pw_epoch", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_iw_epoch", sens_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_iw_epoch", spec_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_pw_epoch", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('test_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.test_accuracy.compute()
        prec = self.test_precision_calculate.compute()
        f1 = self.test_f1.compute()

        self.log('test_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer=='adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        elif self.optimizer=='sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer=='rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.1) # weight_decay=0.001
        elif self.optimizer=='adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0)
        else:
            raise ValueError("Invalid optimizer choice")
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        return [self.optimizer], [lr_scheduler]
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss_epoch"}

class ImageClassifier_unet(pl.LightningModule):
    def __init__(self, model='resnet50', 
                 num_classes=2, 
                 learning_rate=1e-3,
                 weight_decay = 1e-3,
                 optimizer='adam',
                 class_weights=[1,1],
                 pw_based=None,
                 test_pw_based =None,
                 threshold=0.5,
                 rad_img_net=False,
                 hematoma_weights=False,
                 num_features=0, device_use=1):
        
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.pw_based = pw_based
        self.test_pw_based = test_pw_based
        self.class_weights = class_weights
        self.threshold = threshold
        self.rad_img_net = rad_img_net
        self.hematoma_weights = hematoma_weights
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.best_roc_auc = 0
        self.num_features = num_features
        self.device_use = device_use

        # use Unet encoder from paper
        self.backbone = UNetEncoder_paper(in_channels=3)

        # Assign different metrics for training and validation sets to not mix them up
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')

        self.train_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.val_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.test_accuracy_pw = torchmetrics.Accuracy(task='binary')

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

        self.train_f1_pw = torchmetrics.F1Score(task='binary')
        self.val_f1_pw = torchmetrics.F1Score(task='binary')
        self.test_f1_pw = torchmetrics.F1Score(task='binary')

        self.train_precision_calculate = torchmetrics.Precision(task='binary')
        self.val_precision_calculate = torchmetrics.Precision(task='binary')
        self.test_precision_calculate = torchmetrics.Precision(task='binary')

        self.train_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.val_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.test_precision_calculate_pw = torchmetrics.Precision(task='binary')
        
        self.focal_bool = True
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x)
        # logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        # probs = torch.softmax(logits, dim=1)
        # loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        # preds = torch.argmax(logits, dim=1)
        
        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('train_images', grid, self.current_epoch) 

        acc = self.train_accuracy(preds, y)
        prec = self.train_precision_calculate(preds, y)
        f1 = self.train_f1(preds, y)
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}
    
    def training_epoch_end(self, outputs):
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)

            # Just take the second elements of the probabilities and create a new tensor with them. 
            # Second element is the probability of being positive
            if self.focal_bool == True: # only one probality
                concat_single_probs = concat_probs
            else: # two probabilities
                concat_single_probs = torch.tensor([x[1] for x in concat_probs])

            # create a dictionary with patient ids as keys and probabilities as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            # create a dictionary with patient ids as keys and labels as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            # create a dictionary with patient ids as keys and slice numbers as values
            for i in range(len(concat_patient_slice_nums)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            # k: patient_id, v: probability and self.patient_labels[patient_id]: label
            # self.patient_probs = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # combine the probabilities and labels and slice numbers
            # k: patient_id, v: probability and self.patient_labels[patient_id]: label
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # Calculate the epoch metrics for patient level
            acc_pw = self.train_accuracy_pw(y_pred, y_true)
            acc_pw= self.train_accuracy_pw.compute()
            prec_pw = self.train_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.train_precision_calculate_pw.compute()
            f1_pw = self.train_f1_pw(y_pred, y_true)
            f1_pw = self.train_f1_pw.compute()

            self.log("train_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.train_accuracy.compute()
        prec = self.train_precision_calculate.compute()
        f1 = self.train_f1.compute()
        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x) # logits: predicted probablities
        # logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        
        # probs = torch.softmax(logits, dim=1) # softmax: convert logits to probabilities
        # loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        # preds = torch.argmax(logits, dim=1) # predicted class labels 
        acc = self.val_accuracy(preds, y)
        prec = self.val_precision_calculate(preds, y)
        f1 = self.val_f1(preds, y)

        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('val_images', grid, self.current_epoch)
        
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}

    def validation_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_slice_nums = torch.cat(patient_slice_nums, dim=0)

            #just take the second elements of the probabilities and create a new tensor with them
            if self.focal_bool == True: # only one probality
                concat_single_probs = concat_probs
            else:
                concat_single_probs = torch.tensor([x[1] for x in concat_probs])

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                slice_num = concat_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            # self.patient_probs = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # combine the probabilities, labels and slice numbers
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            self.log('val_sens_iw', sens_iw, prog_bar=True, sync_dist=True) # validation sensitivity score
            self.log('val_spec_iw', spec_iw, prog_bar=True, sync_dist=True) # validation specificity score
            self.log('val_roc_auc_iw', roc_auc_iw, prog_bar=True, sync_dist=True) # validation auroc score

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            # Validation AUROC scores
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            self.log('val_roc_auc', roc_auc, prog_bar=True, sync_dist=True) # validation auroc score

            # Calculate the epoch metrics for patient level
            acc_pw = self.val_accuracy_pw(y_pred, y_true)
            acc_pw= self.val_accuracy_pw.compute()
            prec_pw = self.val_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.val_precision_calculate_pw.compute()
            f1_pw = self.val_f1_pw(y_pred, y_true)
            f1_pw = self.val_f1_pw.compute()
            
            self.log("val_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_sens_pw", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_spec_pw", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs] 
            stacked = torch.stack(losses)
            # takes the average of the losses in one epoch
            avg_loss = torch.mean(stacked)
            self.log('val_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.val_accuracy.compute()
        prec = self.val_precision_calculate.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y, patient_id, patient_slice_nums = batch
        logits = self(x)
        # logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))

        # probs = torch.softmax(logits, dim=1) 
        # loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        # preds = torch.argmax(logits, dim=1) 
        acc = self.test_accuracy(preds, y)
        prec = self.test_precision_calculate(preds, y)
        f1 = self.test_f1(preds, y)

        self.log('test_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 
                'probs': probs, 
                'labels': y, 
                'patient_id': patient_id,
                'patient_slice_nums': patient_slice_nums}
    
    def test_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary
        self.patient_probs_with_slices = {} # reset patient probabilities with slice numbers dictionary
    
        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_id'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]

            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_patient_slice_nums = concat_patient_slice_nums + 1 # add 1 to the slice numbers to make them start from 1 instead of 0

            #just take the second elements of the probabilities and create a new tensor with them
            if self.focal_bool == True: # only one probality
                concat_single_probs = concat_probs
            else: # two probabilities
                concat_single_probs = torch.tensor([x[1] for x in concat_probs])

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # create a nested dictionary with patient id as the first key and slice number as the second key and probability as the value
            self.patient_probs_with_slices = {k: {k2: v2 for k2, v2 in zip(self.patient_slice_nums[k], v)} for k, v in self.patient_probs.items()}
            # sort the nested dictionary by slice number
            self.patient_probs_with_slices = {k: dict(sorted(v.items())) for k, v in self.patient_probs_with_slices.items()} 
            print("self.patient_probs_with_slices after sorting: ", self.patient_probs_with_slices)
            #save the nested dictionary as a txt file
            with open('patient_probs_with_slices.txt', 'w') as file:
                file.write(json.dumps(self.patient_probs_with_slices))

            for patient_id in self.patient_probs_with_slices:
                fig = plt.figure()
                plt.scatter(list(self.patient_slice_nums[patient_id]), list(self.patient_probs[patient_id]))
                plt.xlabel("Slice Number")
                plt.ylabel("Probability")
                plt.title("Patient " + str(patient_id))
                plt.savefig(f"plots/{patient_id}.png")
                plt.close(fig)

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            fig = plt.figure()
            plt.scatter(list(concat_labels[concat_labels == 0]), list(concat_single_probs[concat_labels == 0]))
            plt.scatter(list(concat_labels[concat_labels == 1]), list(concat_single_probs[concat_labels == 1]))
            plt.xlabel("Class Label")
            plt.ylabel("Probability")
            plt.title("Patients with Class Label 0 and 1")
            plt.savefig(f"plots/class_label_0_1.png")
            plt.close(fig)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.test_pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.test_pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.test_pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for test_pw_based")
                
            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            self.patient_probs_dict = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            #print assigned means for each patient
            print("Assigned means for each patient: ", self.patient_probs)

            # Calculate the epoch metrics for patient level
            acc_pw = self.test_accuracy_pw(y_pred, y_true)
            acc_pw= self.test_accuracy_pw.compute()
            prec_pw = self.test_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.test_precision_calculate_pw.compute()
            f1_pw = self.test_f1_pw(y_pred, y_true)
            f1_pw = self.test_f1_pw.compute()

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            print(" TEST EPOCH END Predictions PW : ", y_pred)
            print(" TEST EPOCH END Labels PW: ", y_true)

            # compute roc curve 
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            # ROC curve of test data
            score = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) # AUC score. Worst case scneario is 0.5, best case scenario is 1.0
            plt.title('Receiver Operating Characteristic')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            # save the plot
            plt.savefig(f"plots/roc_curve.png")

            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

            self.log('test_roc_auc_score_pw', score, prog_bar=True, sync_dist=True)
            self.log('test_roc_auc_score_iw', roc_auc_iw, prog_bar=True, sync_dist=True)
            self.log("test_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_pw_epoch", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_iw_epoch", sens_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_iw_epoch", spec_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_pw_epoch", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('test_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.test_accuracy.compute()
        prec = self.test_precision_calculate.compute()
        f1 = self.test_f1.compute()

        self.log('test_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer=='adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        elif self.optimizer=='sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer=='rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.001) # weight_decay=0.001
        elif self.optimizer=='adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0)
        else:
            raise ValueError("Invalid optimizer choice")
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        return [self.optimizer], [lr_scheduler]
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss_epoch"}

###### AG UNET MODEL ######

class Attention_block_unet(nn.Module):
    def __init__(self):
        super(Attention_block_unet, self).__init__()
        self.W_g = nn.Sequential( 
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        
        self.W_x = nn.Sequential( # 32,128,128 to a,32,32
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # apply bilinear upsampling to obtain original size -- stated in the paper 
        psi = nn.functional.interpolate(psi, size=(x.size(2), x.size(3)), mode='bilinear') # 
        return x * psi

class AG_UNET(nn.Module):
    # Modified Unet encoder module used in the study "Deep learning for automatically predicting early haematoma expansion in Chinese patients"
    def __init__(self, in_channels=3, out_channels=1):
        super(AG_UNET, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(64)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu_1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.leaky_relu_2 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_conv_1x1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        #add residual connection

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.instance_norm3 = nn.InstanceNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.leaky_relu_3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm4 = nn.InstanceNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.leaky_relu_4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.instance_norm5 = nn.InstanceNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.use_conv_1x1_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.leaky_relu_5 = nn.LeakyReLU(inplace=True)
       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.instance_norm6 = nn.InstanceNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.leaky_relu_6 = nn.LeakyReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm7 = nn.InstanceNorm2d(256)
        self.batch_norm7 = nn.BatchNorm2d(256)
        self.leaky_relu_7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.instance_norm8 = nn.InstanceNorm2d(256)
        self.batch_norm8 = nn.BatchNorm2d(256)
        # add residual connection
        self.leaky_relu_8 = nn.LeakyReLU(inplace=True)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.attention_gate = Attention_block_unet()
        self.instance_norm9 = nn.InstanceNorm2d(512)
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.leaky_relu_9 = nn.LeakyReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm10 = nn.InstanceNorm2d(512)
        self.batch_norm10 = nn.BatchNorm2d(512)
        self.leaky_relu_10 = nn.LeakyReLU(inplace=True)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.instance_norm11 = nn.InstanceNorm2d(512)
        self.batch_norm11 = nn.BatchNorm2d(512)
        self.leaky_relu_11 = nn.LeakyReLU(inplace=True)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, out_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.instance_norm1(out)
        out = self.leaky_relu_1(out)
        out = self.conv2(out)
        out = self.instance_norm2(out)
        out = self.leaky_relu_2(out)
        out = self.pool1(out)

        # apply conv 1x1 
        residual1 = self.use_conv_1x1(out)

        out = self.conv3(out)
        out = self.instance_norm3(out)
        out = self.leaky_relu_3(out)
        out = self.conv4(out)
        out = self.instance_norm4(out)
        out = self.leaky_relu_4(out)
        out = self.conv5(out)
        out = self.instance_norm5(out)
        out = out + residual1
        out = self.leaky_relu_5(out)
        
        out = self.pool2(out)
        # apply conv 1x1
        residual2 = self.use_conv_1x1_2(out)

        out = self.conv6(out)
        out = self.instance_norm6(out)
        out = self.leaky_relu_6(out)
        out = self.conv7(out)
        out = self.instance_norm7(out)
        out = self.leaky_relu_7(out)
        out = self.conv8(out)
        out = self.instance_norm8(out)
        out = out + residual2
        out = self.leaky_relu_8(out)
        
        out = self.conv9(out)
        out = self.instance_norm9(out)
        out = self.leaky_relu_9(out)
        x1 = out

        out = self.conv10(out)
        out = self.instance_norm10(out)
        out = self.leaky_relu_10(out)

        out = self.conv11(out)
        out = self.instance_norm11(out)
        out = self.leaky_relu_11(out)
        g = out

        # attention gate
        att1 = self.attention_gate(g=g, x=x1)
        # concat two outputs (from AG and from FE)
        general_out = torch.cat((att1, g), dim=1)

        out = self.average_pool(general_out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

        # outputs = []
        # for b in range(x.shape[0]): # Go over the number of batch of the input

        #     input = x[b]

        #     out = self.conv1(input)
        #     out = self.instance_norm1(out)
        #     out = self.leaky_relu_1(out)
        #     out = self.conv2(out)
        #     out = self.instance_norm2(out)
        #     out = self.leaky_relu_2(out)
        #     out = self.pool1(out)

        #     # apply conv 1x1 
        #     residual1 = self.use_conv_1x1(out)

        #     out = self.conv3(out)
        #     out = self.instance_norm3(out)
        #     out = self.leaky_relu_3(out)
        #     out = self.conv4(out)
        #     out = self.instance_norm4(out)
        #     out = self.leaky_relu_4(out)
        #     out = self.conv5(out)
        #     out = self.instance_norm5(out)
        #     out = out + residual1
        #     out = self.leaky_relu_5(out)
            
        #     out = self.pool2(out)
        #     # apply conv 1x1
        #     residual2 = self.use_conv_1x1_2(out)

        #     out = self.conv6(out)
        #     out = self.instance_norm6(out)
        #     out = self.leaky_relu_6(out)
        #     out = self.conv7(out)
        #     out = self.instance_norm7(out)
        #     out = self.leaky_relu_7(out)
        #     out = self.conv8(out)
        #     out = self.instance_norm8(out)
        #     out = out + residual2
        #     out = self.leaky_relu_8(out)
            
        #     out = self.conv9(out)
        #     out = self.instance_norm9(out)
        #     out = self.leaky_relu_9(out)
        #     x1 = out

        #     out = self.conv10(out)
        #     out = self.instance_norm10(out)
        #     out = self.leaky_relu_10(out)

        #     out = self.conv11(out)
        #     out = self.instance_norm11(out)
        #     out = self.leaky_relu_11(out)
        #     g = out

        #     # attention gate
        #     att1 = self.attention_gate(g=g.unsqueeze(0), x=x1.unsqueeze(0))

        #     # concat two outputs (from AG and from FE)
        #     general_out = torch.cat((att1.squeeze(0), g), dim=1)

        #     out = self.average_pool(general_out)
        #     out = out.view(-1, out.size(0))
        #     out = self.fc(out)
        #     outputs.append(out)
        # stacked_outputs = torch.stack(outputs, dim=0).squeeze(1).squeeze(1)
        # return stacked_outputs
    
class ImageClassifier_unet_AG(pl.LightningModule):
    def __init__(self, model='resnet50', 
                 num_classes=2, 
                 learning_rate=1e-3,
                 weight_decay = 1e-3,
                 optimizer='adam',
                 class_weights=[1,1],
                 pw_based=None,
                 test_pw_based =None,
                 threshold=0.5,
                 rad_img_net=False,
                 hematoma_weights=False,
                 num_features=0, device_use=1):
        
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.pw_based = pw_based
        self.test_pw_based = test_pw_based
        self.class_weights = class_weights
        self.threshold = threshold
        self.rad_img_net = rad_img_net
        self.hematoma_weights = hematoma_weights
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.best_roc_auc = 0
        self.num_features = num_features
        self.device_use = device_use

        # use Unet encoder from paper
        self.backbone = AG_UNET(in_channels=3)

        # Assign different metrics for training and validation sets to not mix them up
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')

        self.train_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.val_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.test_accuracy_pw = torchmetrics.Accuracy(task='binary')

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

        self.train_f1_pw = torchmetrics.F1Score(task='binary')
        self.val_f1_pw = torchmetrics.F1Score(task='binary')
        self.test_f1_pw = torchmetrics.F1Score(task='binary')

        self.train_precision_calculate = torchmetrics.Precision(task='binary')
        self.val_precision_calculate = torchmetrics.Precision(task='binary')
        self.test_precision_calculate = torchmetrics.Precision(task='binary')

        self.train_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.val_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.test_precision_calculate_pw = torchmetrics.Precision(task='binary')
        
        self.focal_bool = True
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x)
        # logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        # probs = torch.softmax(logits, dim=1)
        # loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        # preds = torch.argmax(logits, dim=1)
        
        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('train_images', grid, self.current_epoch) 

        acc = self.train_accuracy(preds, y)
        prec = self.train_precision_calculate(preds, y)
        f1 = self.train_f1(preds, y)
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}
    
    def training_epoch_end(self, outputs):
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)

            # Just take the second elements of the probabilities and create a new tensor with them. 
            # Second element is the probability of being positive
            if self.focal_bool == True: # only one probality
                concat_single_probs = concat_probs
            else: # two probabilities
                concat_single_probs = torch.tensor([x[1] for x in concat_probs])

            # create a dictionary with patient ids as keys and probabilities as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            # create a dictionary with patient ids as keys and labels as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            # create a dictionary with patient ids as keys and slice numbers as values
            for i in range(len(concat_patient_slice_nums)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            # k: patient_id, v: probability and self.patient_labels[patient_id]: label
            # self.patient_probs = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # combine the probabilities and labels and slice numbers
            # k: patient_id, v: probability and self.patient_labels[patient_id]: label
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # Calculate the epoch metrics for patient level
            acc_pw = self.train_accuracy_pw(y_pred, y_true)
            acc_pw= self.train_accuracy_pw.compute()
            prec_pw = self.train_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.train_precision_calculate_pw.compute()
            f1_pw = self.train_f1_pw(y_pred, y_true)
            f1_pw = self.train_f1_pw.compute()

            self.log("train_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.train_accuracy.compute()
        prec = self.train_precision_calculate.compute()
        f1 = self.train_f1.compute()
        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x) # logits: predicted probablities
        # logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        
        # probs = torch.softmax(logits, dim=1) # softmax: convert logits to probabilities
        # loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        # preds = torch.argmax(logits, dim=1) # predicted class labels 
        acc = self.val_accuracy(preds, y)
        prec = self.val_precision_calculate(preds, y)
        f1 = self.val_f1(preds, y)

        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('val_images', grid, self.current_epoch)
        
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}

    def validation_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_slice_nums = torch.cat(patient_slice_nums, dim=0)

            #just take the second elements of the probabilities and create a new tensor with them
            if self.focal_bool == True: # only one probality
                concat_single_probs = concat_probs
            else:
                concat_single_probs = torch.tensor([x[1] for x in concat_probs])

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                slice_num = concat_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            # self.patient_probs = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # combine the probabilities, labels and slice numbers
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            self.log('val_sens_iw', sens_iw, prog_bar=True, sync_dist=True) # validation sensitivity score
            self.log('val_spec_iw', spec_iw, prog_bar=True, sync_dist=True) # validation specificity score
            self.log('val_roc_auc_iw', roc_auc_iw, prog_bar=True, sync_dist=True) # validation auroc score

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            # Validation AUROC scores
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            self.log('val_roc_auc', roc_auc, prog_bar=True, sync_dist=True) # validation auroc score

            # Calculate the epoch metrics for patient level
            acc_pw = self.val_accuracy_pw(y_pred, y_true)
            acc_pw= self.val_accuracy_pw.compute()
            prec_pw = self.val_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.val_precision_calculate_pw.compute()
            f1_pw = self.val_f1_pw(y_pred, y_true)
            f1_pw = self.val_f1_pw.compute()
            
            self.log("val_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_sens_pw", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_spec_pw", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs] 
            stacked = torch.stack(losses)
            # takes the average of the losses in one epoch
            avg_loss = torch.mean(stacked)
            self.log('val_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.val_accuracy.compute()
        prec = self.val_precision_calculate.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y, patient_id, patient_slice_nums = batch
        logits = self(x)
        # logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))

        # probs = torch.softmax(logits, dim=1) 
        # loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        # preds = torch.argmax(logits, dim=1) 
        acc = self.test_accuracy(preds, y)
        prec = self.test_precision_calculate(preds, y)
        f1 = self.test_f1(preds, y)

        self.log('test_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 
                'probs': probs, 
                'labels': y, 
                'patient_id': patient_id,
                'patient_slice_nums': patient_slice_nums}
    
    def test_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary
        self.patient_probs_with_slices = {} # reset patient probabilities with slice numbers dictionary
    
        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_id'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]

            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_patient_slice_nums = concat_patient_slice_nums + 1 # add 1 to the slice numbers to make them start from 1 instead of 0

            #just take the second elements of the probabilities and create a new tensor with them
            if self.focal_bool == True: # only one probality
                concat_single_probs = concat_probs
            else: # two probabilities
                concat_single_probs = torch.tensor([x[1] for x in concat_probs])

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # create a nested dictionary with patient id as the first key and slice number as the second key and probability as the value
            self.patient_probs_with_slices = {k: {k2: v2 for k2, v2 in zip(self.patient_slice_nums[k], v)} for k, v in self.patient_probs.items()}
            # sort the nested dictionary by slice number
            self.patient_probs_with_slices = {k: dict(sorted(v.items())) for k, v in self.patient_probs_with_slices.items()} 
            print("self.patient_probs_with_slices after sorting: ", self.patient_probs_with_slices)
            #save the nested dictionary as a txt file
            with open('patient_probs_with_slices.txt', 'w') as file:
                file.write(json.dumps(self.patient_probs_with_slices))

            for patient_id in self.patient_probs_with_slices:
                fig = plt.figure()
                plt.scatter(list(self.patient_slice_nums[patient_id]), list(self.patient_probs[patient_id]))
                plt.xlabel("Slice Number")
                plt.ylabel("Probability")
                plt.title("Patient " + str(patient_id))
                plt.savefig(f"plots/{patient_id}.png")
                plt.close(fig)

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            fig = plt.figure()
            plt.scatter(list(concat_labels[concat_labels == 0]), list(concat_single_probs[concat_labels == 0]))
            plt.scatter(list(concat_labels[concat_labels == 1]), list(concat_single_probs[concat_labels == 1]))
            plt.xlabel("Class Label")
            plt.ylabel("Probability")
            plt.title("Patients with Class Label 0 and 1")
            plt.savefig(f"plots/class_label_0_1.png")
            plt.close(fig)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.test_pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.test_pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.test_pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for test_pw_based")
                
            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            self.patient_probs_dict = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            #print assigned means for each patient
            print("Assigned means for each patient: ", self.patient_probs)

            # Calculate the epoch metrics for patient level
            acc_pw = self.test_accuracy_pw(y_pred, y_true)
            acc_pw= self.test_accuracy_pw.compute()
            prec_pw = self.test_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.test_precision_calculate_pw.compute()
            f1_pw = self.test_f1_pw(y_pred, y_true)
            f1_pw = self.test_f1_pw.compute()

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            print(" TEST EPOCH END Predictions PW : ", y_pred)
            print(" TEST EPOCH END Labels PW: ", y_true)

            # compute roc curve 
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            # ROC curve of test data
            score = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) # AUC score. Worst case scneario is 0.5, best case scenario is 1.0
            plt.title('Receiver Operating Characteristic')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            # save the plot
            plt.savefig(f"plots/roc_curve.png")

            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

            self.log('test_roc_auc_score_pw', score, prog_bar=True, sync_dist=True)
            self.log('test_roc_auc_score_iw', roc_auc_iw, prog_bar=True, sync_dist=True)
            self.log("test_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_pw_epoch", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_iw_epoch", sens_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_iw_epoch", spec_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_pw_epoch", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('test_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.test_accuracy.compute()
        prec = self.test_precision_calculate.compute()
        f1 = self.test_f1.compute()

        self.log('test_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer=='adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        elif self.optimizer=='sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer=='rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.001) # weight_decay=0.001
        elif self.optimizer=='adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0)
        else:
            raise ValueError("Invalid optimizer choice")
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        return [self.optimizer], [lr_scheduler]
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss_epoch"}
"""

# removed parameters bacsk and name experment to match with the old model chcekpoint

class ImageClassifier3(pl.LightningModule): # this one is the main one because the actual one is used for the loocv right now. 
    def __init__(self, back, name_experiment, model='eff_s', 
                 num_classes=1, 
                 learning_rate=1e-5,
                 weight_decay = 1e-3,
                 optimizer='adam', 
                 momentum=0.9, 
                 lr_scheduler = None,
                 step_size=15,
                 gamma=0.1,
                 class_weights=[1,1],
                 pw_based="mean",
                 test_pw_based ="mean",
                 threshold=0.5,
                 num_features=0, loss="focal", fold = 0, task = "2d_pred", save_predictions=False):
        
        super().__init__()
        self.name_experiment = name_experiment 
        self.fold = fold
        self.back = back
        self.task = task 
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.pw_based = pw_based
        self.test_pw_based = test_pw_based
        self.class_weights = class_weights
        self.threshold = threshold
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.best_roc_auc = 0
        self.num_features = num_features
        self.save_predictions = save_predictions
        self.gamma = gamma
        self.step_size= step_size

        if self.model == 'resnet18':
            backbone = torchvision.models.resnet18(weights="DEFAULT")
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        if self.model == 'resnet34':
            backbone = torchvision.models.resnet34(weights="DEFAULT")
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        # # train eveything from scratch - Resnet
        if self.model=='resnet50':
            print('Using resnet50')
            backbone = torchvision.models.resnet50(weights="DEFAULT")
            num_features = backbone.fc.in_features
            for name1, param in backbone.named_parameters():
                if 'layer4.1' not in name1: # layer4: 8m 'layer4.1':4m
                    param.requires_grad = False
            self.num_features = num_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone
        
        if self.model=='squeezenet':
            squeezenet = torchvision.models.squeezenet1_0(weights=None)
            squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
            self.backbone = squeezenet

        if self.model=='resnet50_s':
            print('Using resnet50 from scratch')
            backbone = torchvision.models.resnet50(weights=None)
            num_features = backbone.fc.in_features
            self.num_features = num_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        # # swin_t transformer # 15m params
        if self.model=='swin':
            print('Using swin_t')
            model = torchvision.models.swin_t(weights="DEFAULT")
            num_features = model.head.in_features
            for name, param in model.named_parameters():
                if 'features' in name:
                    layer_num = int(name.split('.')[1])  # Get the layer number from the name
                    if layer_num < 6:
                        param.requires_grad = False
            model.head = nn.Linear(num_features, 1)
            self.backbone = model

        if self.model=='swin_s':
            print('Using swin_t from scratch')
            model = torchvision.models.swin_t(weights="DEFAULT")
            num_features = model.head.in_features
            model.head = nn.Linear(num_features, 1)
            self.backbone = model

        # effiecienet b
        if self.model=='eff':
            print('Using eff')
            eff = torchvision.models.efficientnet_b0(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            for name, param in eff.named_parameters():
                if 'features.7' not in name: # features.6: 2M params features.7: 700k 
                    param.requires_grad = False
            self.backbone = eff

        if self.model=='eff_s':
            print('Using eff s')
            eff = torchvision.models.efficientnet_b0(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            self.backbone = eff

        if self.model=='eff_b1':
            print('Using eff b1')
            eff = torchvision.models.efficientnet_b1(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            self.backbone = eff

        if self.model == "eva":
            print("Using eva")
            eva_model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=True, num_classes=1)
            self.backbone = eva_model

        if self.model == "eva_s":
            print("Using eva_s")
            eva_model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=False, num_classes=1)
            self.backbone = eva_model

        if self.model == "r50-vit": # they used this hybrid transformer in Transunet paper (99M parameters)
            print("Using r50-vit")
            r50_vit = timm.create_model('vit_base_r50_s16_384.orig_in21k_ft_in1k', pretrained=True, num_classes=1)
            for name, param in r50_vit.named_parameters(): # just freeze the resnet50 backbone weights but train the vit 
                param.requires_grad = False
            for name, param in r50_vit.blocks[11].named_parameters(): #r50_vit_model.patch_embed # just last layer is open r50_vit_model.blocks[11]
                param.requires_grad = True
            self.backbone = r50_vit

        if self.model == "r50-vit_s": # they used this hybrid transformer in Transunet paper (99M parameters)
            print("Using r50-vit_s")
            r50_vit = timm.create_model('vit_base_r50_s16_384.orig_in21k_ft_in1k', pretrained=True, num_classes=1)
            self.backbone = r50_vit

        if self.model == "nest":
            print("Using nest")
            nest = nest = timm.create_model('timm/nest_tiny_jx.goog_in1k', pretrained=True, num_classes=1)
            self.backbone = nest

        # if self.model == "unet":
        #     print("Using unet")
        #     self.backbone = UNetEncoder_paper(in_channels=3, out_channels=1)

        # if self.model =="ag_unet":
        #     print("Using ag_unet")
        #     self.backbone = AG_UNET(in_channels=3, out_channels=1)

        # if self.model == "unet_2":
        #     print("Using unet_2")
        #     self.backbone = UNetEncoder_paper2(in_channels=3, out_channels=1)

        if self.model == "densenet121":
            print("Using densenet121")
            densenet = torchvision.models.densenet121(pretrained=True)
            num_features = densenet.classifier.in_features
            densenet.classifier = nn.Linear(num_features, 1)
            self.backbone = densenet

        # Assign different metrics for training and validation sets to not mix them up
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')

        self.train_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.val_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.test_accuracy_pw = torchmetrics.Accuracy(task='binary')

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

        self.train_f1_pw = torchmetrics.F1Score(task='binary')
        self.val_f1_pw = torchmetrics.F1Score(task='binary')
        self.test_f1_pw = torchmetrics.F1Score(task='binary')

        self.train_precision_calculate = torchmetrics.Precision(task='binary')
        self.val_precision_calculate = torchmetrics.Precision(task='binary')
        self.test_precision_calculate = torchmetrics.Precision(task='binary')

        self.train_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.val_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.test_precision_calculate_pw = torchmetrics.Precision(task='binary')

        if self.loss=="focal":
            print("Using focal loss")
        else:
            print("Using cross entropy loss")
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x)

        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)
        
        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('train_images', grid, self.current_epoch) 

        acc = self.train_accuracy(preds, y)
        prec = self.train_precision_calculate(preds, y)
        f1 = self.train_f1(preds, y)

        current_learningrate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_learningrate, prog_bar=True, sync_dist=True)

        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}

    def validation_step(self, batch, batch_idx):

        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x) # logits: predicted probablities

        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)
            
        acc = self.val_accuracy(preds, y)
        prec = self.val_precision_calculate(preds, y)
        f1 = self.val_f1(preds, y)

        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('val_images', grid, self.current_epoch)
        
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}
    
    def test_step(self, batch, batch_idx):

        x, y, patient_id, patient_slice_nums = batch
        logits = self(x)
        
        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)

        acc = self.test_accuracy(preds, y)
        prec = self.test_precision_calculate(preds, y)
        f1 = self.test_f1(preds, y)

        self.log('test_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 
                'probs': probs, 
                'labels': y, 
                'patient_id': patient_id,
                'patient_slice_nums': patient_slice_nums}

    def training_epoch_end(self, outputs):
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)

            # Just take the second elements of the probabilities and create a new tensor with them. 
            # Second element is the probability of being positive # only one probality
            concat_single_probs = concat_probs

            # create a dictionary with patient ids as keys and probabilities as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            # create a dictionary with patient ids as keys and labels as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            # create a dictionary with patient ids as keys and slice numbers as values
            for i in range(len(concat_patient_slice_nums)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # Calculate the epoch metrics for patient level
            acc_pw = self.train_accuracy_pw(y_pred, y_true)
            acc_pw= self.train_accuracy_pw.compute()
            prec_pw = self.train_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.train_precision_calculate_pw.compute()
            f1_pw = self.train_f1_pw(y_pred, y_true)
            f1_pw = self.train_f1_pw.compute()

            self.log("train_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.train_accuracy.compute()
        prec = self.train_precision_calculate.compute()
        f1 = self.train_f1.compute()

        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_slice_nums = torch.cat(patient_slice_nums, dim=0)

            #just take the second elements of the probabilities and create a new tensor with them
            # only one probality
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                slice_num = concat_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities, labels and slice numbers
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            self.log('val_sens_iw', sens_iw, prog_bar=True, sync_dist=True) # validation sensitivity score
            self.log('val_spec_iw', spec_iw, prog_bar=True, sync_dist=True) # validation specificity score
            self.log('val_roc_auc_iw', roc_auc_iw, prog_bar=True, sync_dist=True) # validation auroc score

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            # Validation AUROC scores
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            self.log('val_roc_auc', roc_auc, prog_bar=True, sync_dist=True) # validation auroc score

            # Calculate the epoch metrics for patient level
            acc_pw = self.val_accuracy_pw(y_pred, y_true)
            acc_pw= self.val_accuracy_pw.compute()
            prec_pw = self.val_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.val_precision_calculate_pw.compute()
            f1_pw = self.val_f1_pw(y_pred, y_true)
            f1_pw = self.val_f1_pw.compute()
            
            self.log("val_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_sens_pw", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_spec_pw", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs] 
            stacked = torch.stack(losses)
            # takes the average of the losses in one epoch
            avg_loss = torch.mean(stacked)
            self.log('val_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.val_accuracy.compute()
        prec = self.val_precision_calculate.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def test_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary
        self.patient_probs_with_slices = {} # reset patient probabilities with slice numbers dictionary
    
        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_id'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]

            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_patient_slice_nums = concat_patient_slice_nums + 1 # add 1 to the slice numbers to make them start from 1 instead of 0

            #just take the second elements of the probabilities and create a new tensor with them
            # # only one probality
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # create a nested dictionary with patient id as the first key and slice number as the second key and probability as the value
            self.patient_probs_with_slices = {k: {k2: v2 for k2, v2 in zip(self.patient_slice_nums[k], v)} for k, v in self.patient_probs.items()}
            # sort the nested dictionary by slice number
            self.patient_probs_with_slices = {k: dict(sorted(v.items())) for k, v in self.patient_probs_with_slices.items()} 
            print("self.patient_probs_with_slices after sorting: ", self.patient_probs_with_slices)

            # make a dataframe where one column is for patient id, second one is for the average probablities among the slices
            # and the third one is for the label
            self.patient_probs_with_slices_df = pd.DataFrame(columns=["patient_id", "avg_prob", "label"])
            for patient_id in self.patient_probs_with_slices:
               self.patient_probs_with_slices_df = self.patient_probs_with_slices_df.append({"patient_id": patient_id,
                                                                           "avg_prob": np.mean(list(self.patient_probs_with_slices[patient_id].values())),
                                                                          "label": self.patient_labels[patient_id]}, ignore_index=True)
            # create a folder to keep all the test results inside 
            experiments_folder_path =  repo_path + f"/plots/{self.name_experiment}"
            os.makedirs(experiments_folder_path, exist_ok=True)

            #save the nested dictionary as a txt file
            if self.save_predictions:
                with open(experiments_folder_path + "/" + 'patient_probs_with_slices.txt', 'w') as file:
                    file.write(json.dumps(self.patient_probs_with_slices))
                    self.patient_probs_with_slices_df.to_csv(experiments_folder_path + f"/{self.back}_{self.fold}_patient_probs_with_slices_df.csv")

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.test_pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.test_pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.test_pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for test_pw_based")
                
            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            self.patient_probs_dict = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            # Calculate the epoch metrics for patient level
            acc_pw = self.test_accuracy_pw(y_pred, y_true)
            acc_pw= self.test_accuracy_pw.compute()
            prec_pw = self.test_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.test_precision_calculate_pw.compute()
            f1_pw = self.test_f1_pw(y_pred, y_true)
            f1_pw = self.test_f1_pw.compute()

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            print(" TEST EPOCH END Predictions PW : ", y_pred)
            print(" TEST EPOCH END Labels PW: ", y_true)

            # compute roc curve 
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            score = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) # AUC score. Worst case scneario is 0.5, best case scenario is 1.0
            # ROC curve of test data
            if self.save_predictions:
                plt.title(f'Receiver Operating Characteristic-Threshold {self.threshold}')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc='lower right')
                # save the plot
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.savefig(experiments_folder_path + "/" + "roc_curve.png")

            self.log('test_roc_auc_score_pw', score, prog_bar=True, sync_dist=True)
            self.log('test_roc_auc_score_iw', roc_auc_iw, prog_bar=True, sync_dist=True)
            self.log("test_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_pw_epoch", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_iw_epoch", sens_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_iw_epoch", spec_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_pw_epoch", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('test_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # save the patient-wise predictions
        if self.save_predictions:

            # create the predictions folder if it does not exist

            if not os.path.exists(repo_path + f'/predictions'):
                os.makedirs(repo_path + f'/predictions')

            df = pd.DataFrame({'label': y_true.cpu().numpy(), 'pred': y_pred.cpu().numpy()})
            df.to_csv( repo_path + f'/predictions/{self.name_experiment}_{self.back}_{self.fold}_preds.csv', index=False)

        acc = self.test_accuracy.compute()
        prec = self.test_precision_calculate.compute()
        f1 = self.test_f1.compute()

        self.log('test_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer=='adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer=='sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer == "adamw":
            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer choice")
        
        if self.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, verbose=True)
        elif self.lr_scheduler == "reduceonplateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, verbose=True)
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "val_loss_epoch"}
        else:
            self.lr_scheduler = None
            return self.optimizer
        
        return [self.optimizer], [self.lr_scheduler]
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss_epoch"}
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)



class ImageClassifier2(pl.LightningModule): # this one is the main one because the actual one is used for the loocv right now. 
    def __init__(self, model='eff_s', 
                 num_classes=1, 
                 learning_rate=1e-5,
                 weight_decay = 1e-3,
                 optimizer='adam', 
                 momentum=0.9, 
                 lr_scheduler = None,
                 step_size=15,
                 gamma=0.1,
                 class_weights=[1,1],
                 pw_based="mean",
                 test_pw_based ="mean",
                 threshold=0.5,
                 num_features=0, loss="focal", fold = 0, task = "2d_pred", save_predictions=False):
        
        super().__init__()
        # self.name_experiment = name_experiment 
        self.fold = fold
        # self.back = back
        self.task = task 
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.pw_based = pw_based
        self.test_pw_based = test_pw_based
        self.class_weights = class_weights
        self.threshold = threshold
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.best_roc_auc = 0
        self.num_features = num_features
        self.save_predictions = save_predictions
        self.gamma = gamma
        self.step_size= step_size

        if self.model == 'resnet18':
            backbone = torchvision.models.resnet18(weights="DEFAULT")
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        if self.model == 'resnet34':
            backbone = torchvision.models.resnet34(weights="DEFAULT")
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        # # train eveything from scratch - Resnet
        if self.model=='resnet50':
            print('Using resnet50')
            backbone = torchvision.models.resnet50(weights="DEFAULT")
            num_features = backbone.fc.in_features
            for name1, param in backbone.named_parameters():
                if 'layer4.1' not in name1: # layer4: 8m 'layer4.1':4m
                    param.requires_grad = False
            self.num_features = num_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone
        
        if self.model=='squeezenet':
            squeezenet = torchvision.models.squeezenet1_0(weights=None)
            squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
            self.backbone = squeezenet

        if self.model=='resnet50_s':
            print('Using resnet50 from scratch')
            backbone = torchvision.models.resnet50(weights=None)
            num_features = backbone.fc.in_features
            self.num_features = num_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        # # swin_t transformer # 15m params
        if self.model=='swin':
            print('Using swin_t')
            model = torchvision.models.swin_t(weights="DEFAULT")
            num_features = model.head.in_features
            for name, param in model.named_parameters():
                if 'features' in name:
                    layer_num = int(name.split('.')[1])  # Get the layer number from the name
                    if layer_num < 6:
                        param.requires_grad = False
            model.head = nn.Linear(num_features, 1)
            self.backbone = model

        if self.model=='swin_s':
            print('Using swin_t from scratch')
            model = torchvision.models.swin_t(weights="DEFAULT")
            num_features = model.head.in_features
            model.head = nn.Linear(num_features, 1)
            self.backbone = model

        # effiecienet b
        if self.model=='eff':
            print('Using eff')
            eff = torchvision.models.efficientnet_b0(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            for name, param in eff.named_parameters():
                if 'features.7' not in name: # features.6: 2M params features.7: 700k 
                    param.requires_grad = False
            self.backbone = eff

        if self.model=='eff_s':
            print('Using eff s')
            eff = torchvision.models.efficientnet_b0(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            self.backbone = eff

        if self.model=='eff_b1':
            print('Using eff b1')
            eff = torchvision.models.efficientnet_b1(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            self.backbone = eff

        if self.model == "eva":
            print("Using eva")
            eva_model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=True, num_classes=1)
            self.backbone = eva_model

        if self.model == "eva_s":
            print("Using eva_s")
            eva_model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=False, num_classes=1)
            self.backbone = eva_model

        if self.model == "r50-vit": # they used this hybrid transformer in Transunet paper (99M parameters)
            print("Using r50-vit")
            r50_vit = timm.create_model('vit_base_r50_s16_384.orig_in21k_ft_in1k', pretrained=True, num_classes=1)
            for name, param in r50_vit.named_parameters(): # just freeze the resnet50 backbone weights but train the vit 
                param.requires_grad = False
            for name, param in r50_vit.blocks[11].named_parameters(): #r50_vit_model.patch_embed # just last layer is open r50_vit_model.blocks[11]
                param.requires_grad = True
            self.backbone = r50_vit

        if self.model == "r50-vit_s": # they used this hybrid transformer in Transunet paper (99M parameters)
            print("Using r50-vit_s")
            r50_vit = timm.create_model('vit_base_r50_s16_384.orig_in21k_ft_in1k', pretrained=True, num_classes=1)
            self.backbone = r50_vit

        if self.model == "nest":
            print("Using nest")
            nest = nest = timm.create_model('timm/nest_tiny_jx.goog_in1k', pretrained=True, num_classes=1)
            self.backbone = nest

        # if self.model == "unet":
        #     print("Using unet")
        #     self.backbone = UNetEncoder_paper(in_channels=3, out_channels=1)

        # if self.model =="ag_unet":
        #     print("Using ag_unet")
        #     self.backbone = AG_UNET(in_channels=3, out_channels=1)

        # if self.model == "unet_2":
        #     print("Using unet_2")
        #     self.backbone = UNetEncoder_paper2(in_channels=3, out_channels=1)

        if self.model == "densenet121":
            print("Using densenet121")
            densenet = torchvision.models.densenet121(pretrained=True)
            num_features = densenet.classifier.in_features
            densenet.classifier = nn.Linear(num_features, 1)
            self.backbone = densenet

        # Assign different metrics for training and validation sets to not mix them up
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')

        self.train_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.val_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.test_accuracy_pw = torchmetrics.Accuracy(task='binary')

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

        self.train_f1_pw = torchmetrics.F1Score(task='binary')
        self.val_f1_pw = torchmetrics.F1Score(task='binary')
        self.test_f1_pw = torchmetrics.F1Score(task='binary')

        self.train_precision_calculate = torchmetrics.Precision(task='binary')
        self.val_precision_calculate = torchmetrics.Precision(task='binary')
        self.test_precision_calculate = torchmetrics.Precision(task='binary')

        self.train_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.val_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.test_precision_calculate_pw = torchmetrics.Precision(task='binary')

        if self.loss=="focal":
            print("Using focal loss")
        else:
            print("Using cross entropy loss")
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x)

        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)
        
        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('train_images', grid, self.current_epoch) 

        acc = self.train_accuracy(preds, y)
        prec = self.train_precision_calculate(preds, y)
        f1 = self.train_f1(preds, y)

        current_learningrate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_learningrate, prog_bar=True, sync_dist=True)

        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}

    def validation_step(self, batch, batch_idx):

        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x) # logits: predicted probablities

        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)
            
        acc = self.val_accuracy(preds, y)
        prec = self.val_precision_calculate(preds, y)
        f1 = self.val_f1(preds, y)

        # if batch_idx == 0:
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image('val_images', grid, self.current_epoch)
        
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}
    
    def test_step(self, batch, batch_idx):

        x, y, patient_id, patient_slice_nums = batch
        logits = self(x)
        
        if self.loss == "focal":
            logits = logits.squeeze()
            loss = focal_loss(logits, y)
            probs = nn.Sigmoid()(logits)
            preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))
        else:
            logits = logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, y.float())
            probs = nn.Sigmoid()(logits)
            preds = torch.round(probs)

        acc = self.test_accuracy(preds, y)
        prec = self.test_precision_calculate(preds, y)
        f1 = self.test_f1(preds, y)

        self.log('test_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 
                'probs': probs, 
                'labels': y, 
                'patient_id': patient_id,
                'patient_slice_nums': patient_slice_nums}

    def training_epoch_end(self, outputs):
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)

            # Just take the second elements of the probabilities and create a new tensor with them. 
            # Second element is the probability of being positive # only one probality
            concat_single_probs = concat_probs

            # create a dictionary with patient ids as keys and probabilities as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            # create a dictionary with patient ids as keys and labels as values
            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            # create a dictionary with patient ids as keys and slice numbers as values
            for i in range(len(concat_patient_slice_nums)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # Calculate the epoch metrics for patient level
            acc_pw = self.train_accuracy_pw(y_pred, y_true)
            acc_pw= self.train_accuracy_pw.compute()
            prec_pw = self.train_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.train_precision_calculate_pw.compute()
            f1_pw = self.train_f1_pw(y_pred, y_true)
            f1_pw = self.train_f1_pw.compute()

            self.log("train_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.train_accuracy.compute()
        prec = self.train_precision_calculate.compute()
        f1 = self.train_f1.compute()

        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_slice_nums = torch.cat(patient_slice_nums, dim=0)

            #just take the second elements of the probabilities and create a new tensor with them
            # only one probality
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                slice_num = concat_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(slice_num)

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities, labels and slice numbers
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            self.log('val_sens_iw', sens_iw, prog_bar=True, sync_dist=True) # validation sensitivity score
            self.log('val_spec_iw', spec_iw, prog_bar=True, sync_dist=True) # validation specificity score
            self.log('val_roc_auc_iw', roc_auc_iw, prog_bar=True, sync_dist=True) # validation auroc score

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            # Validation AUROC scores
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            self.log('val_roc_auc', roc_auc, prog_bar=True, sync_dist=True) # validation auroc score

            # Calculate the epoch metrics for patient level
            acc_pw = self.val_accuracy_pw(y_pred, y_true)
            acc_pw= self.val_accuracy_pw.compute()
            prec_pw = self.val_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.val_precision_calculate_pw.compute()
            f1_pw = self.val_f1_pw(y_pred, y_true)
            f1_pw = self.val_f1_pw.compute()
            
            self.log("val_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_sens_pw", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_spec_pw", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs] 
            stacked = torch.stack(losses)
            # takes the average of the losses in one epoch
            avg_loss = torch.mean(stacked)
            self.log('val_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.val_accuracy.compute()
        prec = self.val_precision_calculate.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def test_epoch_end(self, outputs):
        print("THE TEST EPOCH END FUNCTION IS CALLED")
        # print list of hyper prameters

        print("Model: ", self.model)
        print("Learning rate: ", self.learning_rate)
        print("Weight decay: ", self.weight_decay)
        print("Optimizer: ", self.optimizer)
        print("Momentum: ", self.momentum)
        print("Loss: ", self.loss)
        print("PW based: ", self.pw_based)
        print("Test PW based: ", self.test_pw_based)
        print("Threshold: ", self.threshold)
        print("Class weights: ", self.class_weights)
        print("Fold: ", self.fold)
        print("Task: ", self.task)
        print("Save predictions: ", self.save_predictions)
        
        # calculate metrics for the whole epoch
        self.patient_probs = {} # reset patient probabilities dictionary
        self.patient_labels = {} # reset patient labels dictionary
        self.patient_slice_nums = {} # reset patient slice numbers dictionary
        self.patient_probs_with_slices = {} # reset patient probabilities with slice numbers dictionary
    
        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_id'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]

            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_patient_slice_nums = concat_patient_slice_nums + 1 # add 1 to the slice numbers to make them start from 1 instead of 0

            #just take the second elements of the probabilities and create a new tensor with them
            # # only one probality
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # create a nested dictionary with patient id as the first key and slice number as the second key and probability as the value
            self.patient_probs_with_slices = {k: {k2: v2 for k2, v2 in zip(self.patient_slice_nums[k], v)} for k, v in self.patient_probs.items()}
            # sort the nested dictionary by slice number
            self.patient_probs_with_slices = {k: dict(sorted(v.items())) for k, v in self.patient_probs_with_slices.items()} 
            print("self.patient_probs_with_slices after sorting: ", self.patient_probs_with_slices)

            # make a dataframe where one column is for patient id, second one is for the average probablities among the slices
            # and the third one is for the label
            self.patient_probs_with_slices_df = pd.DataFrame(columns=["patient_id", "avg_prob", "label"])
            for patient_id in self.patient_probs_with_slices:
               self.patient_probs_with_slices_df = self.patient_probs_with_slices_df.append({"patient_id": patient_id,
                                                                           "avg_prob": np.mean(list(self.patient_probs_with_slices[patient_id].values())),
                                                                          "label": self.patient_labels[patient_id]}, ignore_index=True)
            # # create a folder to keep all the test results inside 
            # experiments_folder_path =  repo_path + f"/plots/{self.name_experiment}"
            # os.makedirs(experiments_folder_path, exist_ok=True)

            # #save the nested dictionary as a txt file
            # if self.save_predictions:
            #     with open(experiments_folder_path + "/" + 'patient_probs_with_slices.txt', 'w') as file:
            #         file.write(json.dumps(self.patient_probs_with_slices))
            #         self.patient_probs_with_slices_df.to_csv(experiments_folder_path + f"/{self.back}_{self.fold}_patient_probs_with_slices_df.csv")

            # convert them to cpu
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.test_pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.test_pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.test_pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for test_pw_based")
                
            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            self.patient_probs_dict = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            # Calculate the epoch metrics for patient level
            acc_pw = self.test_accuracy_pw(y_pred, y_true)
            acc_pw= self.test_accuracy_pw.compute()
            prec_pw = self.test_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.test_precision_calculate_pw.compute()
            f1_pw = self.test_f1_pw(y_pred, y_true)
            f1_pw = self.test_f1_pw.compute()

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)

            print(" TEST EPOCH END Predictions PW : ", y_pred)
            print(" TEST EPOCH END Labels PW: ", y_true)

            # compute roc curve 
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            score = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) # AUC score. Worst case scneario is 0.5, best case scenario is 1.0
            # ROC curve of test data
            self.log('test_roc_auc_score_pw', score, prog_bar=True, sync_dist=True)
            self.log('test_roc_auc_score_iw', roc_auc_iw, prog_bar=True, sync_dist=True)
            self.log("test_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_pw_epoch", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_iw_epoch", sens_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_iw_epoch", spec_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_pw_epoch", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('test_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # # save the patient-wise predictions
        # if self.save_predictions:
        #     df = pd.DataFrame({'label': y_true.cpu().numpy(), 'pred': y_pred.cpu().numpy()})
        #     df.to_csv( repo_path + f'/predictions/{self.name_experiment}_{self.back}_{self.fold}_preds.csv', index=False)

        acc = self.test_accuracy.compute()
        prec = self.test_precision_calculate.compute()
        f1 = self.test_f1.compute()

        self.log('test_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer=='adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer=='sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer == "adamw":
            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer choice")
        
        if self.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, verbose=True)
        elif self.lr_scheduler == "reduceonplateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, verbose=True)
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "val_loss_epoch"}
        else:
            self.lr_scheduler = None
            return self.optimizer
        
        return [self.optimizer], [self.lr_scheduler]
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss_epoch"}
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)