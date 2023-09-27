# The implementation code is based on the paper:
# Young-Jin Heo, Young-Ju Choi, Young-Woon Lee, and Byung-Gyu Kim. Deepfake detection scheme based on vision transformer and distillation, 2021. 3
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import json
DEFAULTS = {
    "network": "dpn",
    "encoder": "dpn92",
    "model_params": {},
    "optimizer": {
        "batch_size": 32,
        "type": "SGD",  # supported: SGD, Adam
        "momentum": 0.9,
        "weight_decay": 0,
        "clip": 1.,
        "learning_rate": 0.1,
        "classifier_lr": -1,
        "nesterov": True,
        "schedule": {
            "type": "constant",  # supported: constant, step, multistep, exponential, linear, poly
            "mode": "epoch",  # supported: epoch, step
            "epochs": 10,
            "params": {}
        }
    },
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config


from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


class LRStepScheduler(_LRScheduler):
    def __init__(self, optimizer, steps, last_epoch=-1):
        self.lr_steps = steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        pos = max(bisect_right([x for x, y in self.lr_steps], self.last_epoch) - 1, 0)
        return [self.lr_steps[pos][1] if self.lr_steps[pos][0] <= self.last_epoch else base_lr for base_lr in self.base_lrs]


class PolyLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to poly learning rate policy
    """
    def __init__(self, optimizer, max_iter=90000, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch = (self.last_epoch + 1) % self.max_iter
        return [base_lr * ((1 - float(self.last_epoch) / self.max_iter) ** (self.power)) for base_lr in self.base_lrs]

class ExponentialLRScheduler(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= 0:
            return self.base_lrs
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]
    
    
    
    
import cv2
# ! pip install apex
# from apex.optimizers import FusedAdam, FusedSGD
from timm.optim import AdamW
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.rmsprop import RMSprop
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(optimizer_config, model, master_params=None):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    if optimizer_config.get("classifier_lr", -1) != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if k.find("encoder") != -1:
                net_params.append(v)
            else:
                classifier_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        if master_params:
            params = master_params
        else:
            params = model.parameters()

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "FusedSGD":
        optimizer = FusedSGD(params,
                             lr=optimizer_config["learning_rate"],
                             momentum=optimizer_config["momentum"],
                             weight_decay=optimizer_config["weight_decay"],
                             nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "FusedAdam":
        optimizer = FusedAdam(params,
                              lr=optimizer_config["learning_rate"],
                              weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "AdamW":
        optimizer = AdamW(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "RmsProp":
        optimizer = RMSprop(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = LRStepScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "clr":
        scheduler = CyclicLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = ExponentialLRScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "poly":
        scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":
        def linear_lr(it):
            return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler



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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.xception import xception
#from models.efficientnet import EfficientNet
import kornia
import torchvision.models as torchm
#cont_gradfrom utils import cont_grad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from functools import partial
import numpy as np
import torch
from timm.models import skresnext50_32x4d
from timm.models.dpn import dpn92, dpn131
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, \
    vit_base_patch16_384, vit_base_patch32_384, vit_large_patch16_224, vit_large_patch16_384, \
    vit_large_patch32_384 #, vit_huge_patch16_224, vit_huge_patch32_384 update
from timm.models.senet import legacy_seresnext50_32x4d
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
#from taming_transformer import Decoder, VUNet, ActNorm
import functools
#from vit_pytorch.distill import DistillableViT, DistillWrapper, DistillableEfficientViT
import re
import argparse
import json
import os
from collections import defaultdict

from sklearn.metrics import log_loss
from torch import topk


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


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        base_model = timm.create_model("resnet18", pretrained=True)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, 1)
        self.features = base_model

    def forward(self, x):
        x = self.features(x)
        return x
# resnet18 = ResNet18()

encoder_params = {

    "vit_large_patch32_384": {
        "features": 1024,
        "init_op": partial(vit_large_patch32_384, pretrained=True, drop_path_rate=0.2)
    },
    "vit_large_patch16_384": {
        "features": 1024,
        "init_op": partial(vit_large_patch16_384, pretrained=True, drop_path_rate=0.2)
    },
    "vit_large_patch16_224": {
        "features": 1024,
        "init_op": partial(vit_large_patch16_224, pretrained=True, drop_path_rate=0.2)
    },
    "vit_base_patch_16_384": {
        "features": 768,
        "init_op": partial(vit_base_patch16_384, pretrained=True, drop_path_rate=0.2)
    },
    "vit_base_patch_32_384": {
        "features": 768,
        "init_op": partial(vit_base_patch32_384, pretrained=True, drop_path_rate=0.2)
    },
    "dpn92": {
        "features": 2688,
        "init_op": partial(dpn92, pretrained=True)
    },
    "dpn131": {
        "features": 2688,
        "init_op": partial(dpn131, pretrained=True)
    },
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns_03d": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_03d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_04d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns_04d": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.4)
    },
    "se50": {
        "features": 2048,
        "init_op": partial(legacy_seresnext50_32x4d, pretrained=True)
    },
    "sk50": {
        "features": 2048,
        "init_op": partial(skresnext50_32x4d, pretrained=True)
    },
}




class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: int, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x


class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    
class DeepFakeClassifier_Distill(nn.Module):
    def __init__(self, dropout_rate=0.0) -> None:
        super().__init__()
        resnet18 = ResNet18()
        self.encoder = resnet18
        self.backbone = resnet18
        self.teacher = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        self.temperature = 1.
        self.alpha = 0.5
        self.bce = nn.BCEWithLogitsLoss()
        self.avg_pool = AdaptiveAvgPool2d((None,1024))

        for p in self.teacher.parameters():
            p.requires_grad = False

        
    def forward(self, x): #eye
        teacher_logits = self.teacher(x)
        ##print(teacher_logits.shape)
        student_logits = self.backbone(x)
        ##print(student_logits.shape)
        distill_logits = self.encoder(x)
        ###print(distill_logits.shape)
            
        return student_logits, distill_logits, teacher_logits #, teacher_logits#student_logits, distill_logits, teacher_logits #loss * alpha + distill_loss * (1 - alpha)


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
    
    

