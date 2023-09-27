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


class SN_FromFineTuned(nn.Module):
    def __init__(self, fine_tuned_model):
        super(SN_FromFineTuned, self).__init__()

        self.SN_features = nn.Sequential(*list(fine_tuned_model.features.children())[:5])
        
    def forward(self, x):
        x = self.SN_features(x)
        return x
    
class DN_FromFineTuned(nn.Module):
    def __init__(self, fine_tuned_model):
        super(DN_FromFineTuned, self).__init__()

        self.DN_features = nn.Sequential(*list(fine_tuned_model.features.children())[5:])
        
    def forward(self, x):
        x = self.DN_features(x)
        return x


class Diffusion_Network(nn.Module):
    def __init__(self, in_channels, iterations=10):
        super(Diffusion_Network, self).__init__()
        
        self.in_channels = in_channels
        self.iterations = iterations
        self.diffusion_layers = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False) for _ in range(iterations)]
        )
  
    def forward(self, x):
            df_states = []
            for state in range(self.iterations):
                x = nn.ReLU()(self.diffusion_layers[state](x))
                df_states.append(x)
            return df_states
        


class WholeNetwork(nn.Module):
    def __init__(self, SN, DiffN, DN):
        super(WholeNetwork, self).__init__()

        # Explicitly set requires_grad for the encoder's parameters
        for param in SN.parameters():
            param.requires_grad = True
        for param in DiffN.parameters():
            param.requires_grad = True
        for param in DN.parameters():
            param.requires_grad = True

        self.SN = SN
        self.DiffN = DiffN
        self.DN = DN  
        
    def forward(self, x):

        # compute diffusion and transient states of diffusion
        x0 = self.SN(x)

        # Compute diffusion and transient states of diffusion using the diffused feature map
        df_states = self.DiffN(x0)

        # average to create a more robust diffused-transient-representation
        df = x0
        for df_s in df_states:
            df = df + df_s
        df = df/(len(df_states)+1)
        
        df = torch.cat([df], dim=1) # format now 64*148*148

        p = self.DN(df)
        
        return p





# # -----------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------

# transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ])
    
# dataset = Dataset_Builder("MIXED/FAKE/",
#                           "MIXED/REAL/",
#                           transform=transform)
# print("\n\n")
# print("-------------- compute - time ----------------------------------------------------------------------\n")
# import time
# # i want to access inference comp time in sec
# ft_initial_network = Initial_Network().cuda()
# SN = SN_FromFineTuned(ft_initial_network)
# DN = DN_FromFineTuned(ft_initial_network)
# Ν = 5
# DiffN = Diffusion_Network(in_channels = 64, iterations=Ν)
# whole_net = WholeNetwork(SN, DiffN, DN).cuda()



# dataiter = iter(DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, worker_init_fn=_init_fn))
# image, _, _ = next(dataiter)
# image = image.cuda()

# with torch.no_grad():
#     ft_initial_network.eval()
#     whole_net.eval()
#     # that the first forward pass might include some overhead due to caching operations
#     output1 = ft_initial_network(image)
#     output2 = whole_net(image)

#     # so measure for fair now
#     # # Measure time for ft_initial_network inference
#     start_time = time.time()
#     output1 = ft_initial_network(image)
#     end_time = time.time()
#     print(f"Inference time for ft_initial_network: {end_time - start_time:.4f} seconds")

#     # Measure time for whole_net inference
#     start_time = time.time()
#     output2 = whole_net(image)
#     end_time = time.time()
#     print(f"Inference time for whole_net: {end_time - start_time:.4f} seconds")
# print("\n\n")
# # # -------------- comp - time ----------------------------------------------------------------------
# # time_overhead = (0.0140-0.0131)/0.0131


# # -----------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------

# s = 1