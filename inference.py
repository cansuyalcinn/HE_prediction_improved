import os
from sklearn.utils.class_weight import compute_class_weight
import sys; sys.path.insert(0, os.path.abspath("../"))
from dataset import *
from utils import *
from model import *
from model import ImageClassifier2
import torch.utils.data as data
import random
import argparse
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

# chcekpoint for the best da_sy(5) model
checkpoint_path = "/home/cansu/HE_classification_synthesis/checkpoints/threshold50/org_da_sy_es5_hf05_repeat_2711/best_model.ckpt"

model = ImageClassifier2.load_from_checkpoint(checkpoint_path)
model.eval().cuda(device=1)

unseen_neg_samples = pd.read_csv("/home/cansu/HE_classification_synthesis/notebooks/not_sampled_negative_52_cases.csv")

X_train =[]
y_train = []
X_val = []
X_test = []
y_val = []
y_test = []

md_path = "/home/cansu/HE_classification_synthesis/data/metadata.csv" # contains all the 205 cases

for i in tqdm(range(len(unseen_neg_samples))):
    # add the index numbers of the unseen negative samples to the X_test list
    X_test.append(unseen_neg_samples["index"].tolist()[i])
    y_test.append(unseen_neg_samples["label"].tolist()[i])

indexes = X_train, X_val, X_test, y_train, y_val, y_test

dm = HEPredDataModule(split_indexes=indexes, 
                        filter_slices=True,
                        mask=True, 
                        batch_size=1, 
                        num_workers=8, 
                        use_2d=True, 
                        return_type='image',
                        under_sampling=False,
                        over_sampling=False,
                        threshold=0.5,
                        md_path=md_path,
                        basal_fu=False,
                        roi = False, problem = "prediction",
                        image_size = 512,
                        roi_size=512, lesion=False,
                        test_type = "t5",
                        apply_hflip = False,
                        apply_affine = False,
                        apply_gaussian_blur= False,
                        affine_degree=10,
                        affine_translate=0,
                        affine_scale=1.0,
                        affine_shear=0, 
                        hflip_p = 0.5, affine_p = 0.5,)
dm.setup()

# Read the test dataset from the datamodule

test_loader = dm.test_dataloader()

img_list = []
label_list = []
id_list = []
snum_list = []

for batch in test_loader:
    image, label, id, snum = batch
    img_list.append(image)
    label_list.append(label)
    id_list.append(id)
    snum_list.append(snum)

# with torch.no_grad():
#     y_hat = model(image.cuda(device=1))
#     print("y_hat",y_hat)
#     print("label",label)
#     print("id",id)
#     print("snum",snum)