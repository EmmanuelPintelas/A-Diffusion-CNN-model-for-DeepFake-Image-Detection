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


# based on the idea of paper: Hsu, C. C., Zhuang, Y. X., & Lee, C. Y. (2020). Deep fake image detection based on pairwise learning. Applied Sciences, 10(1), 370.

# The key idea here is that the Siamese network is being used to learn a feature space where similar images are close together 
# and dissimilar images are far apart. Once this feature space is learned, 
# a classifier can be trained to predict the individual labels of each image based on their location in this feature space.



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


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        base_model = timm.create_model("resnet18", pretrained=True)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, num_features//2)
        self.base_network = base_model
        
    def forward_one(self, x):
        return self.base_network(x)
        
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + 
                           (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


class CFF_and_Classifier(nn.Module):
    def __init__(self, siamese):
        super(CFF_and_Classifier, self).__init__()

        self.CFF = nn.Sequential(*list(siamese.base_network.children())[:-3])
        
        conv_classifier = timm.create_model("resnet18", pretrained=True)
        conv_classifier.fc = nn.Linear(conv_classifier.fc.in_features, 1)
        conv_classifier = nn.Sequential(*list(conv_classifier.children())[-3:])
        self.Classifier = conv_classifier
        
    def forward(self, x):
        x = self.CFF(x)
        x = self.Classifier(x)
        return x



class Dataset_Pairwise(Dataset):
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
        image1 = Image.open(image_path)
        if self.transform:
            image1 = self.transform(image1)

        idx2 = random.randint(0, len(self.all_images) - 1)
        image2_path = self.all_images[idx2]
        image2 = Image.open(image2_path)
        if self.transform:
            image2 = self.transform(image2)

        # Determine pair label
        if (self.labels[idx] == 1 and self.labels[idx2] == 1) or (self.labels[idx] == 0 and self.labels[idx2] == 0):
            pair_label = 1
        else:
            pair_label = 0

        return image1, image2, pair_label


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
    
def show_batch(images1, images2, pair_labels):
    fig = plt.figure(figsize=(10, 10))
    
    for i in range(images1.shape[0]):
        plt.subplot(4, 4, i * 2 + 1)
        imshow(images1[i])
        plt.subplot(4, 4, i * 2 + 2)
        imshow(images2[i])
        plt.title(f'Pair Label: {pair_labels[i].item()}')
    
    plt.tight_layout()
    plt.show()

transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
])

dataset_pairs = Dataset_Pairwise("EXP/FAKE/",
                          "EXP/REAL/",
                          transform=transform)

loader_pairs = DataLoader(dataset_pairs, batch_size=8, shuffle=False, num_workers=0, worker_init_fn=_init_fn)


# --- check ---
dataiter = iter(loader_pairs)
images1, images2, pair_labels = next(dataiter)
images_to_plot1, images_to_plot2 = np.transpose(images1.numpy(), (0, 2, 3, 1)), np.transpose(images2.numpy(), (0, 2, 3, 1))
#R-R or F-F -> 1, R-F -> 0
show_batch(images_to_plot1, images_to_plot2, pair_labels)

