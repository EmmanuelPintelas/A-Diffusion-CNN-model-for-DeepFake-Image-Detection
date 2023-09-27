import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pickle
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import timm
import random
import math
import torch.utils.model_zoo as model_zoo
##from torch.nn import init
import logging
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import gc

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




class AttentionMap(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask',torch.zeros([1,1,24,24]))
        self.mask[0,0,2:-2,2:-2]=1
        self.num_attentions=out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1) #extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.num_attentions==0:
            return torch.ones([x.shape[0],1,1,1],device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)+1
        mask=F.interpolate(self.mask,(x.shape[2],x.shape[3]),mode='nearest')
        return x*mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, attentions,norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions=F.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            feature_matrix=torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix=torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm==1:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            feature_matrix = F.normalize(feature_matrix,p=2,dim=-1)
        if norm==3:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix

class Texture_Enhance_v2(nn.Module):
    def __init__(self,num_features,num_attentions):
        super().__init__()
        self.output_features=num_features
        self.output_features_d=num_features
        self.conv_extract=nn.Conv2d(num_features,num_features,3,padding=1)
        self.conv0=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,5,padding=2,groups=num_attentions)
        self.conv1=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn1=nn.BatchNorm2d(num_features*num_attentions)
        self.conv2=nn.Conv2d(num_features*2*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn2=nn.BatchNorm2d(2*num_features*num_attentions)
        self.conv3=nn.Conv2d(num_features*3*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn3=nn.BatchNorm2d(3*num_features*num_attentions)
        self.conv_last=nn.Conv2d(num_features*4*num_attentions,num_features*num_attentions,1,groups=num_attentions)
        self.bn4=nn.BatchNorm2d(4*num_features*num_attentions)
        self.bn_last=nn.BatchNorm2d(num_features*num_attentions)
        
        self.M=num_attentions
    def cat(self,a,b):
        B,C,H,W=a.shape
        c=torch.cat([a.reshape(B,self.M,-1,H,W),b.reshape(B,self.M,-1,H,W)],dim=2).reshape(B,-1,H,W)
        return c

    def forward(self,feature_maps,attention_maps=(1,1)):
        B,N,H,W=feature_maps.shape
        if type(attention_maps)==tuple:
            attention_size=(int(H*attention_maps[0]),int(W*attention_maps[1]))
        else:
            attention_size=(attention_maps.shape[2],attention_maps.shape[3])
        feature_maps=self.conv_extract(feature_maps)
        feature_maps_d=F.adaptive_avg_pool2d(feature_maps,attention_size)
        if feature_maps.size(2)>feature_maps_d.size(2):
            feature_maps=feature_maps-F.interpolate(feature_maps_d,(feature_maps.shape[2],feature_maps.shape[3]),mode='nearest')
        attention_maps=(torch.tanh(F.interpolate(attention_maps.detach(),(H,W),mode='bilinear',align_corners=True))).unsqueeze(2) if type(attention_maps)!=tuple else 1
        feature_maps=feature_maps.unsqueeze(1)
        feature_maps=(feature_maps*attention_maps).reshape(B,-1,H,W)
        feature_maps0=self.conv0(feature_maps)
        feature_maps1=self.conv1(F.relu(self.bn1(feature_maps0),inplace=True))
        feature_maps1_=self.cat(feature_maps0,feature_maps1)
        feature_maps2=self.conv2(F.relu(self.bn2(feature_maps1_),inplace=True))
        feature_maps2_=self.cat(feature_maps1_,feature_maps2)
        feature_maps3=self.conv3(F.relu(self.bn3(feature_maps2_),inplace=True))
        feature_maps3_=self.cat(feature_maps2_,feature_maps3)
        feature_maps=F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_),inplace=True))),inplace=True)
        feature_maps=feature_maps.reshape(B,-1,N,H,W)
        return feature_maps,feature_maps_d

        
class Auxiliary_Loss_v2(nn.Module):
    def __init__(self,M,N,C,alpha=0.05,margin=1,inner_margin=[0.1,5]):
        super().__init__()
        self.register_buffer('feature_centers',torch.zeros(M,N))
        self.register_buffer('alpha',torch.tensor(alpha))
        self.num_classes=C
        self.margin=margin
        self.atp=AttentionPooling()
        self.register_buffer('inner_margin',torch.Tensor(inner_margin))

    def forward(self,feature_map_d,attentions,y):
        B,N,H,W=feature_map_d.size()
        B,M,AH,AW=attentions.size()
        if AH!=H or AW!=W:
            attentions=F.interpolate(attentions,(H,W),mode='bilinear',align_corners=True)
        feature_matrix=self.atp(feature_map_d,attentions)
        feature_centers=self.feature_centers
        center_momentum=feature_matrix-feature_centers
        real_mask=(y==0).view(-1,1,1)
        fcts=self.alpha*torch.mean(center_momentum*real_mask,dim=0)+feature_centers
        fctsd=fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd,torch.distributed.ReduceOp.SUM)
                    fctsd/=torch.distributed.get_world_size()
                self.feature_centers=fctsd  
        inner_margin=self.inner_margin[y]
        intra_class_loss=F.relu(torch.norm(feature_matrix-fcts,dim=[1,2])*torch.sign(inner_margin)-inner_margin)
        intra_class_loss=torch.mean(intra_class_loss)
        inter_class_loss=0
        for j in range(M):
            for k in range(j+1,M):
                inter_class_loss+=F.relu(self.margin-torch.dist(fcts[j],fcts[k]),inplace=False)
        inter_class_loss=inter_class_loss/M/self.alpha
        #fmd=attentions.flatten(2)
        #diverse_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss+inter_class_loss,feature_matrix

    
class Initial_Network(nn.Module):
    def __init__(self, base_model_name="resnet18"):
        super(Initial_Network, self).__init__()

        base_model = timm.create_model(base_model_name, pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


    
class MAT(nn.Module):
    def __init__(self, attention_layer='final', num_classes=2, M=8, mid_dims=256,\
                 dropout_rate=0.5, drop_final_rate=0.5, pretrained=False, alpha=0.05, size=(600, 600), margin=1, inner_margin=[0.01, 0.02]):
        super(MAT, self).__init__()
        self.num_classes = 2
        self.M = M
        self.net = Initial_Network().features.to(device)
        self.feature_layer = nn.Sequential(*list(self.net.children())[:6])  
        self.attention_layer = attention_layer
        
        with torch.no_grad():
            layers = self.net(torch.zeros(1,3,size[0],size[1]).to(device))
        num_features = layers.shape[1]
        self.mid_dims=mid_dims
        if pretrained:
            a=torch.load(pretrained,map_location='cpu')
            keys={i:a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys=a['state_dict']
            self.net.load_state_dict(keys,strict=False)
        self.attentions = AttentionMap(layers.shape[1], self.M)
        self.atp=AttentionPooling()
        self.texture_enhance=Texture_Enhance_v2(num_features,M)
        self.num_features=self.texture_enhance.output_features
        self.num_features_d=self.texture_enhance.output_features_d
        self.projection_local=nn.Sequential(nn.Linear(M*self.num_features,mid_dims),nn.Hardswish(),nn.Linear(mid_dims,mid_dims))
        self.project_final = nn.Linear(layers.shape[1], mid_dims)
        self.ensemble_classifier_fc=nn.Sequential(nn.Linear(mid_dims*2,mid_dims),nn.Hardswish(),nn.Linear(mid_dims,num_classes))
        self.auxiliary_loss=Auxiliary_Loss_v2(M,self.num_features_d,num_classes,alpha,margin,inner_margin)
        self.dropout=nn.Dropout1d(dropout_rate,inplace=True)
        self.dropout_final=nn.Dropout(drop_final_rate,inplace=True)
        #self.center_loss=Center_Loss(self.num_features*M,num_classes)

    def train_batch(self,x,y,jump_aux=False,drop_final=False):
        layers = self.net(x)
        if self.feature_layer=='logits':
            logits=layers['logits']
            loss=F.cross_entropy(logits,y)
            return dict(loss=loss,logits=logits)
        feature_maps = layers[self.feature_layer]
        raw_attentions = layers[self.attention_layer]
        attention_maps_=self.attentions(raw_attentions)
        dropout_mask=self.dropout(torch.ones([attention_maps_.shape[0],self.M,1],device=x.device))
        attention_maps=attention_maps_*torch.unsqueeze(dropout_mask,-1)
        feature_maps,feature_maps_d=self.texture_enhance(feature_maps,attention_maps_)
        feature_maps_d=feature_maps_d-feature_maps_d.mean(dim=[2,3],keepdim=True)
        feature_maps_d=feature_maps_d/(torch.std(feature_maps_d,dim=[2,3],keepdim=True)+1e-8)
        feature_matrix_=self.atp(feature_maps,attention_maps_)
        feature_matrix=feature_matrix_*dropout_mask

        B,M,N = feature_matrix.size()
        if not jump_aux:
            aux_loss,feature_matrix_d=self.auxiliary_loss(feature_maps_d,attention_maps_,y)
        else:
            feature_matrix_d=self.atp(feature_maps_d,attention_maps_)
            aux_loss=0
        feature_matrix=feature_matrix.view(B,-1)
        feature_matrix=F.hardswish(self.projection_local(feature_matrix))
        final=layers['final']
        attention_maps=attention_maps.sum(dim=1,keepdim=True)
        final=self.atp(final,attention_maps,norm=1).squeeze(1)
        final=self.dropout_final(final)
        projected_final=F.hardswish(self.project_final(final))
        #projected_final=self.dropout(projected_final.view(B,1,-1)).view(B,-1)
        if drop_final:
            projected_final*=0
        feature_matrix=torch.cat((feature_matrix,projected_final),1)
        ensemble_logit=self.ensemble_classifier_fc(feature_matrix)
        ensemble_loss=F.cross_entropy(ensemble_logit,y)
        return dict(ensemble_loss=ensemble_loss,aux_loss=aux_loss,attention_maps=attention_maps_,ensemble_logit=ensemble_logit,feature_matrix=feature_matrix_,feature_matrix_d=feature_matrix_d)


    def forward(self, x):
        layers = self.net(x)
        raw_attentions = layers
        attention_maps = self.attentions(raw_attentions)
        feature_maps = layers
        feature_maps, feature_maps_d = self.texture_enhance(feature_maps, attention_maps)
        
        feature_matrix=self.atp(feature_maps,attention_maps)
        B,M,N = feature_matrix.size()
        feature_matrix=self.dropout(feature_matrix)
        feature_matrix=feature_matrix.view(B,-1)
        feature_matrix=F.hardswish(self.projection_local(feature_matrix))
        final = layers
        attention_maps2=attention_maps.sum(dim=1,keepdim=True)
        final=self.atp(final,attention_maps2,norm=1).squeeze(1)
        projected_final=F.hardswish(self.project_final(final))
        feature_matrix=torch.cat((feature_matrix,projected_final),1)
        ensemble_logit=self.ensemble_classifier_fc(feature_matrix)
        return ensemble_logit

def load_state(net,ckpt):
    sd=net.state_dict()
    nd={}
    for i in ckpt:
        if i in sd and sd[i].shape==ckpt[i].shape:
            nd[i]=ckpt[i]
    net.load_state_dict(nd,strict=False)
    
mat = MAT()
mat

