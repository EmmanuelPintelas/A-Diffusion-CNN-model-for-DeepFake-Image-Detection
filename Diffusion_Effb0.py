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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# --------------- MANDATORY ---------------
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


class ExpertsDataset(Dataset):
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
    def __init__(self, base_model_name="efficientnet_b0"):
        super(Initial_Network, self).__init__()

        base_model = timm.create_model(base_model_name, pretrained=True)
        num_features = base_model.classifier.in_features
        base_model.classifier = nn.Linear(num_features, 1)
        self.features = base_model

    def forward(self, x):
        x = self.features(x)
        return x



class SN_FromFineTuned(nn.Module):
    def __init__(self, fine_tuned_model):
        super(SN_FromFineTuned, self).__init__()

        self.SN_features = nn.Sequential(*list(fine_tuned_model.features.children())[:1])


    def forward(self, x):
        x = self.SN_features(x)
        return x

class DN_FromFineTuned(nn.Module):
    def __init__(self, fine_tuned_model):
        super(DN_FromFineTuned, self).__init__()

        self.DN_features = nn.Sequential(*list(fine_tuned_model.features.children())[1:])

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

        # average to create a more robust diffused-transient-dynamics-representation
        df = x0
        for df_s in df_states:
            df = df + df_s
        df = df/(len(df_states)+1)

        df = torch.cat([df], dim=1) # format now 64*148*148

        p = self.DN(df)

        return p





class Trainer:
    def __init__(self, model, device, patience=10, lr = 0.0001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4) # optim.Adadelta(model.parameters(), lr=1.0) #
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.7, patience=patience//2, verbose=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.patience = patience
        self.best_val_loss = float('inf')
        self.best_val_acc = float(0)
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
        for epoch in range(epochs):


            train_loss, train_acc = self.train(train_loader)
            val_loss, val_acc, auc, sen, spe = self.validate(val_loader)



            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                ##torch.save(self.model.state_dict(), 'best_model.pth')
                best_param = pickle.dumps(self.model.state_dict()) 
                self.epochs_without_improvement = 0

                print(f"Epoch {epoch+1}/{epochs}")
                print("-"*15)
                print(f"Train Loss: {train_loss:.3f} Acc: {train_acc:.2f}%")
                print(f"val Acc: {val_acc:.2f}% val AUC: {auc:.4f} val Sensitivity: {sen:.4f} val Specificity: {spe:.4f}")

                t_loss, t_acc, t_auc, t_sen, t_spe = self.validate(test_loader)
                print(f"test Acc: {t_acc:.2f}% test AUC: {t_auc:.4f} test Sensitivity: {t_sen:.4f} test Specificity: {t_spe:.4f}")

            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement > self.patience:
                    print("Early stopping")
                    break

            self.scheduler.step(val_acc)
        ##self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.load_state_dict(pickle.loads(best_param))








device = torch.device("cuda")

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
dataset = ExpertsDataset("MIXED/FAKE/",
                          "MIXED/REAL/",
                          transform=transform)









# # -------------- comp - time ----------------------------------------------------------------------
# print("\n\n")
# print("-------------- compute - time ----------------------------------------------------------------------\n")
# import time
# # i want to access inference comp time in sec
# ft_initial_network = Initial_Network().cuda()
# SN = SN_FromFineTuned(ft_initial_network)
# DN = DN_FromFineTuned(ft_initial_network)
# Ν = 5
# DiffN = Diffusion_Network(in_channels = 32, iterations=Ν)
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

# s = 1








torch.manual_seed(SEED)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# ---------------------

torch.manual_seed(SEED)
val_size = int(0.1 * train_size)
train_dataset, val_dataset = random_split(train_dataset, [train_size-val_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, worker_init_fn=_init_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0, worker_init_fn=_init_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



torch.manual_seed(SEED)
#----------------------------Our model------------------------------------
learning_rate = 0.0001
ft_initial_network = Initial_Network()
trainer = Trainer(ft_initial_network, device, lr = learning_rate)
trainer.run(train_loader, val_loader, test_loader, epochs = 15)
SN = SN_FromFineTuned(ft_initial_network)
DN = DN_FromFineTuned(ft_initial_network)
Ν = 5
DiffN = Diffusion_Network(in_channels = 32, iterations=Ν)
whole_net = WholeNetwork(SN, DiffN, DN)
trainer = Trainer(whole_net, device, lr = learning_rate)
trainer.run(train_loader, val_loader, test_loader, epochs = 45)
#----------------------------------------------------------------

