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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# 1. Define the custom Dataset class
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


class CNN(nn.Module):
    def __init__(self, base_model_name="resnet18"):
        super(CNN, self).__init__()

        base_model = timm.create_model(base_model_name, pretrained=True)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, 1)
        self.features = base_model

    def forward(self, x):
        x = self.features(x)
        return x


