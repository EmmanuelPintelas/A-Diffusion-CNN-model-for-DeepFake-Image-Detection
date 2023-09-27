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


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention_map = self.sigmoid(self.conv(x))
        return x * attention_map

class AttentionSiameseNetwork(nn.Module):
    def __init__(self):
        super(AttentionSiameseNetwork, self).__init__()
        
        base_model = timm.create_model("resnet18", pretrained=True)
        num_features = base_model.fc.in_features
        self.num_features = num_features
        base_model.fc = nn.Linear(num_features, num_features//2)
        
        # add attention module after the 4th layer (layer1)
        attention_module = AttentionModule(in_channels=64)
        base_model.layer1 = nn.Sequential(base_model.layer1, attention_module)
        
        self.base_network = base_model
        
    def forward_one(self, x):
        return self.base_network(x)
        
    def forward(self, Ia, Ip, In):
        Oa = self.forward_one(Ia)
        Op = self.forward_one(Ip)
        On = self.forward_one(In)
        return Oa, Op, On


class Deepfake_Classifier(nn.Module):
    def __init__(self, at_si_net):
        super(Deepfake_Classifier, self).__init__()

        self.cnn = at_si_net.base_network
        self.num_features = at_si_net.num_features
        self.fc = nn.Linear(self.num_features//2, 1)

    def forward(self, x):

        x = self.cnn(x)
        x = self.fc(x)

        return x


class Triplet_Loss(nn.Module):
    def __init__(self, margin=1.0):
        super(Triplet_Loss, self).__init__()
        self.margin = margin

    def forward(self, Oa, Op, On):
        # Compute the pairwise distances between the embeddings
        delta_positive = torch.norm(Oa - Op, p=2)
        delta_negative = torch.norm(Oa - On, p=2)

        # Compute the triplet loss
        loss = torch.relu(self.margin + delta_positive - delta_negative)

        return loss



class Dataset_Triplet(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None):
        self.transform = transform
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.all_images = self.fake_images + self.real_images
        self.labels = [1] * len(self.fake_images) + [0] * len(self.real_images)  # 1 for fake, 0 for real

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_a = Image.open(self.all_images[idx])
        if self.transform:
            image_a = self.transform(image_a)

        id_f = random.randint(0, len(self.fake_images) - 1)
        image_f = Image.open(self.fake_images[id_f])
        if self.transform:
            image_f = self.transform(image_f)
            
        id_r = random.randint(0, len(self.real_images) - 1)
        image_r = Image.open(self.real_images[id_r])
        if self.transform:
            image_r = self.transform(image_r)

        return image_a, image_f, image_r


class Dataset_Classification(Dataset):
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



def imshow(img):
    plt.imshow(img)
    plt.axis('off')
    
def show_batch(images_to_plot_a, images_to_plot_f, images_to_plot_r):
    batch_size = len(images_to_plot_a)
    fig, axs = plt.subplots(batch_size, 3, figsize=(10, 20))

    for i in range(batch_size):
        axs[i, 0].imshow(images_to_plot_a[i])
        axs[i, 0].axis('off')
        axs[i, 1].imshow(images_to_plot_f[i])
        axs[i, 1].axis('off')
        axs[i, 2].imshow(images_to_plot_r[i])
        axs[i, 2].axis('off')

    plt.show()

transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor()
])

dataset_triplets = Dataset_Triplet("EXP/FAKE/",
                          "EXP/REAL/",
                          transform=transform)

loader_triplets = DataLoader(dataset_triplets, batch_size=8, shuffle=True, num_workers=0, worker_init_fn=_init_fn)


# --- check ---
dataiter = iter(loader_triplets)
image_a, image_f, image_r = next(dataiter)
# all torch.Size([8, 3, 600, 600]) 8 = bsize
images_to_plot_a, images_to_plot_f, images_to_plot_r = np.transpose(image_a.numpy(), (0, 2, 3, 1)), np.transpose(image_f.numpy(), (0, 2, 3, 1)), np.transpose(image_r.numpy(), (0, 2, 3, 1))
#R-R or F-F -> 1, R-F -> 0
show_batch(images_to_plot_a, images_to_plot_f, images_to_plot_r) # i want to plot like triplets for the batch
