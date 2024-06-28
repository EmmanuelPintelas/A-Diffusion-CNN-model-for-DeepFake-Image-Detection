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

from Diffusion_CNN import *



# --------------- MANDATORY (global seeding for reproducable results) ---------------
SEED = 0
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



class Trainer:
    def __init__(self, model, device, patience=10, lr = 0.0001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.7, patience=5, verbose=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.patience = patience
        self.best_val_acc = float(0)
        self.best_param = pickle.dumps(self.model.state_dict())
        self.epochs_without_improvement = 0


    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        corrects = 0
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, targets = data.to(self.device), target.float().unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(output))
            corrects += (preds == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * corrects / len(train_loader.dataset)
        return avg_loss, accuracy
                
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        corrects = 0
        all_preds = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, targets = data.to(self.device), target.float().unsqueeze(1).to(self.device)
                
                output = self.model(data)
                loss = self.loss_fn(output, targets)
                val_loss += loss.item()

                probs = output 

                preds = torch.round(torch.sigmoid(output))
                corrects += (preds == targets).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())


        avg_loss = val_loss / len(val_loader.dataset)
        accuracy = 100. * corrects / len(val_loader.dataset)
        auc = roc_auc_score(all_targets, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return avg_loss, accuracy, auc, sensitivity, specificity
    
    
    def run(self, train_loader, val_loader, test_loader, epochs):
        global global_best_val_acc, global_best_param, f_t_acc, f_t_auc, f_t_sen, f_t_spe
        for epoch in range(epochs):
            train_loss, train_acc = self.train(train_loader)
            _, acc, auc, sen, spe = self.validate(val_loader)
            
            if acc > self.best_val_acc:
                self.best_val_acc = acc
                self.best_param = pickle.dumps(self.model.state_dict()) 
                self.epochs_without_improvement = 0
                
                print(f"Epoch {epoch+1}/{epochs}")
                print("-"*15)
                print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
                print(f"val Acc: {acc:.2f}% val AUC: {auc:.4f} val Sensitivity: {sen:.4f} val Specificity: {spe:.4f}")
                
                _, t_acc, t_auc, t_sen, t_spe = self.validate(test_loader)
                print(f"test Acc: {t_acc:.2f}% test AUC: {t_auc:.4f} test Sensitivity: {t_sen:.4f} test Specificity: {t_spe:.4f}")
                
                if acc >= global_best_val_acc:
                    global_best_val_acc = acc
                    global_best_param = pickle.dumps(self.model.state_dict())
                    f_t_acc, f_t_auc, f_t_sen, f_t_spe = t_acc, t_auc, t_sen, t_spe

                
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement > self.patience:
                    print("Early stopping")
                    break
            
            self.scheduler.step(acc)

        self.model.load_state_dict(pickle.loads(self.best_param))




def main_10_fold_evaluate(dataset_name):

    device = torch.device("cuda")

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

        torch.manual_seed(SEED)
        learning_rate = 0.0001
        backbone = Initial_Network()
        print('Fine-Tuning BN') 
        trainer = Trainer(backbone, device, lr = learning_rate)
        trainer.run(train_loader, val_loader, test_loader, epochs = 15)
        SN = SN_FromFineTuned(backbone)
        DN = DN_FromFineTuned(backbone)
        Ν = 5
        DiffN = Diffusion_Network(in_channels = 64, iterations=Ν)
        whole_net = WholeNetwork(SN, DiffN, DN)
        print('\nFine-Tuning WN') 
        trainer = Trainer(whole_net, device, lr = learning_rate)
        trainer.run(train_loader, val_loader, test_loader, epochs = 45) #


        # Testing Evaluation of the best model in the validation set
        print(f"\n\ntesting results of the best model identified for the validation set of Fold {fold+1}/{n_splits}\n")
        print(f"test Acc: {f_t_acc:.2f}% \ntest AUC: {f_t_auc:.4f} \ntest Sensitivity: {f_t_sen:.4f} \ntest Specificity: {f_t_spe:.4f}\n\n")

        folds_acc.append(f_t_acc)
        folds_auc.append(f_t_auc)
        folds_sen.append(f_t_sen)
        folds_spe.append(f_t_spe)

    print('\n\n------' + str(dataset_name) + '---FINAL 10-cross Testing Results \n')
    print(f"test Acc: {np.mean(folds_acc):.2f}% \ntest AUC: {np.mean(folds_auc):.4f} \ntest Sensitivity: {np.mean(folds_sen):.4f} \ntest Specificity: {np.mean(folds_spe):.4f}\n\n")
        


main_10_fold_evaluate('EXP')
main_10_fold_evaluate('DFDC')
main_10_fold_evaluate('DFMNIST')
main_10_fold_evaluate('MIXED')







