import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pickle
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import timm
import random
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import copy
from copy import deepcopy


# --------------- MANDATORY (global seeding for reproducable results) ---------------
SEED = 1
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)    
torch.manual_seed(SEED)
np.random.seed(SEED)

def _init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# -----------------------------------------




# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-Libraries_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
 
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from   sklearn                   import tree
from sklearn.tree import export_text

import pandas as pd
import random as rn
import numpy as np

import math
from math import e

from sklearn.linear_model import LogisticRegression


import sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from   sklearn                   import tree
from sklearn.tree import export_text
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from scipy.stats import entropy
from scipy.stats import norm, kurtosis
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import gc
from sklearn.model_selection import train_test_split
from numpy import save
from numpy import load
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC


import random as rn
import random
import math
from math import e
import os


import cv2
import PIL
from PIL import Image 
import argparse
import random as rng
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline



def pipeline_LR(X_train, y_train):
        estimators = [
            ('scaler', MinMaxScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ]
        pipe = Pipeline(estimators)
        pipe.fit(X_train, y_train)
        return pipe


def evaluate (clf, feat, labels):

        preds =  clf.predict (feat)
        probs =  clf.predict_proba (feat)[:,1]

        corrects = (preds == labels)
        accuracy = 100. * len(corrects[corrects==True]) / len(labels)
        auc = roc_auc_score(labels, probs)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return accuracy, auc, sensitivity, specificity


def Resize(B_IMAGES, s_ize):
        B_IMAGES_Resized = []
        for b_image in B_IMAGES:
            b_im = cv2.resize(b_image,(s_ize, s_ize))
            B_IMAGES_Resized.append(b_im)
        return np.array(B_IMAGES_Resized)



def Standarize (data):
    fff = data
    # # # #---------------------------> Standarize
    for i in range(len(fff[0])):
                fff[:, i] = (fff[:, i] - np.mean(fff[:,i]))/np.std(fff[:,i])
    ii = np.argwhere(np.isnan(fff))
    for i,j in ii:
            fff[i,j] = 0
    data= fff
    return data





class Initial_Network(nn.Module):
    def __init__(self, base_model_name="resnet18"):
        super(Initial_Network, self).__init__()

        base_model = timm.create_model(base_model_name, pretrained=True)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, 1)
        self.features = base_model

    def forward(self, x):
        x = self.features(x)
        return x

class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(*list(backbone.features.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x






class Dataset_Builder(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None):
        self.transform = transform
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.all_images = self.fake_images + self.real_images
        self.labels = [1] * len(self.fake_images) + [0] * len(self.real_images)  # 1 for fake, 0 for real

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        
        name_id = image_path.split('/')[-1].replace('.jpg','')
        return image, label, name_id





def multi_view_feature_extractor (loader, mode, backbone):

    device = torch.device("cuda")

    def transformations(images): # assume images normalized in 0-1
        images_uint8 = (images * 255).byte() 
        images_uint8 = images_uint8.permute(0, 2, 3, 1).numpy()
        
        images_view0 = np.copy(images_uint8)# 256 init
        images_view1 = np.copy(Resize(images_uint8, 180))
        images_view2 = np.copy(Resize(images_uint8, 300))
        images_view3 = np.copy(images_uint8)
        #images_view3[:,:,:,0],images_view3[:,:,:,1],images_view3[:,:,:,2] = images_uint8[:,:,:,2]*images_uint8[:,:,:,0], images_uint8[:,:,:,0]*images_uint8[:,:,:,1], images_uint8[:,:,:,1]*images_uint8[:,:,:,2]
        images_view3[:,:,:,0],images_view3[:,:,:,1],images_view3[:,:,:,2] = images_uint8[:,:,:,0]/((0.03)*images_uint8[:,:,:,1]), images_uint8[:,:,:,1]/((0.03)*images_uint8[:,:,:,2]), images_uint8[:,:,:,2]/((0.03)*images_uint8[:,:,:,0])


        images_view0 = torch.from_numpy(images_view0).permute(0, 3, 1, 2)
        images_view0 = images_view0.float() / 255.0
        images_view1 = torch.from_numpy(images_view1).permute(0, 3, 1, 2)
        images_view1 = images_view1.float() / 255.0
        images_view2 = torch.from_numpy(images_view2).permute(0, 3, 1, 2)
        images_view2 = images_view2.float() / 255.0
        images_view3 = torch.from_numpy(images_view3).permute(0, 3, 1, 2)
        images_view3 = images_view3.float() / 255.0

        return images_view0, images_view1, images_view2, images_view3

    def train_1epoch (net, opt, view, labels):
        net.train()     
        opt.zero_grad()
        output = net(view.to(device))
        loss = loss_fn(output, labels.float().unsqueeze(1).to(device))
        loss.backward()
        opt.step()


    if mode == 'train':
        backbone = [deepcopy(Initial_Network()).to(device) for _ in range(4)]
        optimizer = [Adam(backbone[_].parameters(), lr=0.0001, weight_decay=1e-4) for _ in range(4)]
        loss_fn = nn.BCEWithLogitsLoss()

        for images, labels, _ in loader:
            views = transformations(images) 
            for i in range (len(views)):
                train_1epoch (backbone[i], optimizer[i], views[i], labels)


    FEATURES, LABELS = [], []
    for images, labels, _ in loader:
        images_view0, images_view1, images_view2, images_view3 = transformations(images)

        FE0 = Encoder(backbone[0]).cuda().eval()
        with torch.no_grad():
            feat0 = FE0(images_view0.cuda())
        feat0 = feat0.cpu().numpy()

        FE1 = Encoder(backbone[1]).cuda().eval()
        with torch.no_grad():
            feat1 = FE1(images_view1.cuda())
        feat1 = feat1.cpu().numpy()

        FE2 = Encoder(backbone[2]).cuda().eval()
        with torch.no_grad():
            feat2 = FE2(images_view2.cuda())
        feat2 = feat2.cpu().numpy()

        FE3 = Encoder(backbone[3]).cuda().eval()
        with torch.no_grad():
            feat3 = FE3(images_view3.cuda())
        feat3 = feat3.cpu().numpy()

        feat = np.concatenate((feat0,feat1,feat2,feat3),axis=1)

        FEATURES.append(feat)
        LABELS.append(labels)

    FEATURES = np.concatenate(FEATURES)
    #FEATURES = Standarize (FEATURES)
    LABELS = np.concatenate(LABELS)

    if mode == 'train':
         return FEATURES, LABELS, backbone
    else:
        return FEATURES, LABELS






def main_10_fold_evaluate(dataset_name):

    transform = transforms.Compose([
                transforms.Resize((256, 256)), 
                transforms.ToTensor()])


    if dataset_name == 'EXP':
            dataset = Dataset_Builder("EXP/FAKE/",
                                    "EXP/REAL/",
                                    transform=transform)
    elif dataset_name == 'DFDC':
            dataset = Dataset_Builder("DFDC/FAKE/",
                                    "DFDC/REAL/",
                                    transform=transform)
    elif dataset_name == 'DFMNIST':
            dataset = Dataset_Builder("DFMNIST/FAKE/",
                                    "DFMNIST/REAL/",
                                    transform=transform)
    elif dataset_name == 'MIXED':
            dataset = Dataset_Builder("MIXED/FAKE/",
                                    "MIXED/REAL/",
                                    transform=transform)  


    n_splits = 10
    torch.manual_seed(SEED)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


    folds_acc, folds_auc, folds_sen, folds_spe = [], [], [], []
        
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            global global_best_val_acc, global_best_param, f_t_acc, f_t_auc, f_t_sen, f_t_spe
            global_best_val_acc, global_best_param, f_t_acc, f_t_auc, f_t_sen, f_t_spe = float(0), float(0), float(0), float(0), float(0), float(0)
            print(f'Fold {fold+1}/{n_splits}')
            train_dataset = Subset(dataset, train_idx)
            test_dataset = Subset(dataset, test_idx)

            train_size = len(train_dataset)
            val_size = int(0.1 * train_size)

            torch.manual_seed(SEED)
            train_dataset, val_dataset = random_split(train_dataset, [train_size-val_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, worker_init_fn=_init_fn)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, worker_init_fn=_init_fn)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            train_feat, train_labels, backbone = multi_view_feature_extractor(train_loader, 'train', None)
            val_feat, val_labels = multi_view_feature_extractor(val_loader,'test', backbone)
            test_feat, test_labels = multi_view_feature_extractor(test_loader,'test', backbone)


            clf = pipeline_LR(np.array(train_feat), np.array(train_labels)) #fit
            v = evaluate (clf, val_feat, val_labels)
            t = evaluate (clf, test_feat, test_labels)
            f_t_acc, f_t_auc, f_t_sen, f_t_spe = t
            print(f"\n\ntesting results of the best model identified for the validation set of Fold {fold+1}/{n_splits}\n")
            print(f"test Acc: {f_t_acc:.2f}% \ntest AUC: {f_t_auc:.3f} \ntest Sensitivity: {f_t_sen:.3f} \ntest Specificity: {f_t_spe:.3f}\n\n")
            folds_acc.append(f_t_acc)
            folds_auc.append(f_t_auc)
            folds_sen.append(f_t_sen)
            folds_spe.append(f_t_spe)

    print('\n\n------FINAL 10-cross Testing Results \n')
    print(f"test Acc: {np.mean(folds_acc):.2f}% \ntest AUC: {np.mean(folds_auc):.4f} \ntest Sensitivity: {np.mean(folds_sen):.4f} \ntest Specificity: {np.mean(folds_spe):.4f}\n\n")
            


##for _ in range (2):



