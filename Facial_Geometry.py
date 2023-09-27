
# based on the idea of paper: Liang, P., Liu, G., Xiong, Z., Fan, H., Zhu, H., & Zhang, X. (2023). 
# A facial geometry based detection model for face manipulation using CNN-LSTM architecture. Information Sciences, 633, 370-383.


import torch
import timm
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
import cv2
import random
import math
from torch.nn import init
import logging
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import re
import argparse
import json
from collections import defaultdict
from sklearn.metrics import log_loss

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



class Image_Landmarks_Dataset(Dataset):
    def __init__(self, root_dir_fake, root_dir_real, transform=None):
        self.transform = transform
        
        self.root_dir_fake = root_dir_fake
        self.root_dir_real = root_dir_real
        
        self.folders_fake = [d for d in os.listdir(root_dir_fake) if os.path.isdir(os.path.join(root_dir_fake, d))]
        self.image_paths_fake = []
        self.landmark_paths_fake = []
        
        self.folders_real = [d for d in os.listdir(root_dir_real) if os.path.isdir(os.path.join(root_dir_real, d))]
        self.image_paths_real = []
        self.landmark_paths_real = []
        
        for folder in self.folders_fake:
                    folder_path = os.path.join(root_dir_fake, folder)

                    all_files = os.listdir(folder_path)

                    # Filter out image and landmark files
                    img_files = [f for f in all_files if f.endswith('.jpg')]
                    landmark_files = [f for f in all_files if f.endswith('_landmarks.pkl')]

                    self.image_paths_fake.append(os.path.join(folder_path, img_files[0]))
                    self.landmark_paths_fake.append(os.path.join(folder_path, landmark_files[0]))
                    
        for folder in self.folders_real:
                    folder_path = os.path.join(root_dir_real, folder)

                    all_files = os.listdir(folder_path)

                    # Filter out image and landmark files
                    img_files = [f for f in all_files if f.endswith('.jpg')]
                    landmark_files = [f for f in all_files if f.endswith('_landmarks.pkl')]

                    self.image_paths_real.append(os.path.join(folder_path, img_files[0]))
                    self.landmark_paths_real.append(os.path.join(folder_path, landmark_files[0]))
                    
        self.all_image_paths = self.image_paths_fake + self.image_paths_real    
        self.all_landmark_paths = self.landmark_paths_fake + self.landmark_paths_real
        
        self.labels = [1] * len(self.image_paths_fake) + [0] * len(self.image_paths_real)  # 1 for fake, 0 for real
        
        

    def __len__(self):
        #print(self.image_paths)
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        # Load image
        img_name = self.all_image_paths[idx]
        image = Image.open(img_name)

        # Load landmarks
        landmark_name = self.all_landmark_paths[idx]
        with open(landmark_name, 'rb') as f:
            landmarks = pickle.load(f)
        
        landmarks = {k: torch.tensor(np.mean(v, axis=0)) for k, v in landmarks.items()}

        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]

        return image, landmarks, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
])

# Define the datasets
root_dir = '/kaggle/input/fork-of-landmarks-creation-exp'
image_landmarks_dataset = Image_Landmarks_Dataset(os.path.join(root_dir, 'Folder_Fake'), os.path.join(root_dir, 'Folder_Real'), transform=transform)
image_landmarks_dataset_loader = DataLoader(image_landmarks_dataset, batch_size=5, shuffle=True)

# check
dataiter = iter(image_landmarks_dataset_loader)
images, landmarks, labels  = next(dataiter)
print(landmarks['LEy']) # left eye landmarks bathces (of given batch_size)


dataiter = iter(image_landmarks_dataset_loader)
images, landmarks, labels  = next(dataiter)
print(landmarks['LEy']) # left eye landmarks bathces (of given batch_size)

device = torch.device("cuda")

# Define the transformations: resizing and normalization
transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ImageNet data
])

torch.manual_seed(SEED)
train_size = int(0.9 * len(image_landmarks_dataset))
test_size = len(image_landmarks_dataset) - train_size
train_dataset, test_dataset = random_split(image_landmarks_dataset, [train_size, test_size])
# ---------------------

torch.manual_seed(SEED)
val_size = int(0.1 * train_size)
train_dataset, val_dataset = random_split(train_dataset, [train_size-val_size, val_size])
    
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, worker_init_fn=_init_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0, worker_init_fn=_init_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# check
images, landmarks, labels  = next(iter(train_loader))
print(landmarks['LEy'][-3:]) # left eye landmarks bathces (of given batch_size)
images, landmarks, labels  = next(iter(val_loader))
print(landmarks['LEy'][-3:]) # left eye landmarks bathces (of given batch_size)
images, landmarks, labels  = next(iter(test_loader))
print(landmarks['LEy'][-3:]) # left eye landmarks bathces (of given batch_size)



class FGPM(nn.Module):
    #facial geometry features eaxtraction
    #landmarks regions extracted based on: 
    # https://www.kaggle.com/code/emmanuelpintelas/fork-of-landmarks-creation-exp
    
    def __init__(self):
        self.landmarks_keys = list(["LEy","LEb","REy","REb","N","M"])
        self.landmark_pairs = [(self.landmarks_keys[i], self.landmarks_keys[j]) for i in range(len(self.landmarks_keys)) for j in range(i+1, len(self.landmarks_keys))]
        self.landmark_triplets = [(self.landmarks_keys[i], self.landmarks_keys[j], self.landmarks_keys[k]) for i in range(len(self.landmarks_keys)) for j in range(i+1, len(self.landmarks_keys)) for k in range(j+1, len(self.landmarks_keys))]
        super(FGPM, self).__init__()

    def Facial_Geometry_Features(self, region_landmarks):
        def compute_distance(point1, point2):
            return torch.norm(point1 - point2, dim=1)

        def compute_angle(point1, point2, point3):
            vector1 = point1 - point2
            vector2 = point3 - point2
            cosine_angle = torch.sum(vector1 * vector2, dim=1) / (torch.norm(vector1, dim=1) * torch.norm(vector2, dim=1))
            angle = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0))
            return torch.rad2deg(angle)

        def compute_area(point1, point2, point3):
            return 0.5 * torch.abs(point1[:, 0] * (point2[:, 1] - point3[:, 1]) + point2[:, 0] * (point3[:, 1] - point1[:, 1]) + point3[:, 0] * (point1[:, 1] - point2[:, 1]))


        features = []
        for pair in self.landmark_pairs:
            key1, key2 = pair
            landmark1 = region_landmarks[key1]
            landmark2 = region_landmarks[key2]
            distance = compute_distance(landmark1.to(device), landmark2.to(device))
            features.append(distance)

        for triplet in self.landmark_triplets:
            key1, key2, key3 = triplet
            landmark1 = region_landmarks[key1]
            landmark2 = region_landmarks[key2]
            landmark3 = region_landmarks[key3]
            angle = compute_angle(landmark1.to(device), landmark2.to(device), landmark3.to(device))
            features.append(angle)

        for triplet in self.landmark_triplets:
            key1, key2, key3 = triplet
            landmark1 = region_landmarks[key1]
            landmark2 = region_landmarks[key2]
            landmark3 = region_landmarks[key3]
            area = compute_area(landmark1.to(device), landmark2.to(device), landmark3.to(device))
            features.append(area)

        features = torch.stack(features, dim=1)
        return features.to(device)


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        base_model = timm.create_model("resnet18", pretrained=True)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        num_features = base_model.fc.in_features

        self.lstm = nn.LSTM(num_features//2, num_features//2, batch_first=True)
        self.fc2 = nn.Linear(num_features//2, 1)
        self.num_features = num_features
  

    def forward(self, x):
        # Pass through convolutional layers
        x = self.cnn(x)

        # Reshape for LSTM
        x = x.view(-1, 1, self.num_features//2)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Pass through final fully connected layer
        x = self.fc2(x[:,0])
        ##x = self.fc2(x)

        return x

    
    
class DeepFakeDetector(nn.Module):
    def __init__(self, cnn_lstm, fgpm):
        super(DeepFakeDetector, self).__init__()
        self.fgpm = fgpm
        self.cnn = cnn_lstm.cnn
        self.lstm = cnn_lstm.lstm # an do an de douleyei, bazw cnn features
        self.num_features = cnn_lstm.num_features

        self.classifier = nn.Linear(55 + self.num_features, 1)

    def forward(self, x, landmarks):

        fgpm_out = self.fgpm.Facial_Geometry_Features(landmarks)

        x = self.cnn(x)
        ##print(x.shape)
        #x = x.view(-1, 1, 128)
        #cnn_lstm_out, _ = self.lstm(x)

        fgpm_out = fgpm_out.to(torch.float32)
        ##fgpm_out = (fgpm_out - fgpm_out.min()) / (fgpm_out.max() - fgpm_out.min())
        fgpm_out = (fgpm_out - fgpm_out.mean(dim=0)) / fgpm_out.std(dim=0)


        #cnn_lstm_out = cnn_lstm_out[:,0].to(torch.float32)

        # fgpm_out = (fgpm_out - fgpm_out.mean()) / fgpm_out.std()
        #cnn_lstm_out = (cnn_lstm_out - cnn_lstm_out.mean(dim=0)) / cnn_lstm_out.std(dim=0)

        #print(fgpm_out)
        #print(cnn_lstm_out)

        fused = torch.cat((fgpm_out, x), dim=1)


        ##print(fused.shape)

        p = self.classifier(fused)

        return p

# cnn_lstm = CNN_LSTM()
# train cnn_lstm
# after training:
# deepfake_model = DeepFakeDetector (cnn_lstm, fgpm)

# train deepfake_model
# images, landmarks = batch
# p = deepfake_model(images, landmarks)



global global_best_val_acc, global_best_param, f_t_acc, f_t_auc, f_t_sen, f_t_spe
global_best_val_acc = float(0)

class Trainer:
    def __init__(self, model, device, patience=15, lr = 0.0001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4) # optim.Adadelta(model.parameters(), lr=1.0) #
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.7, patience=patience//2, verbose=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.patience = patience
        self.best_val_acc = float(0)
        self.best_param = pickle.dumps(self.model.state_dict())
        self.epochs_without_improvement = 0
        
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        corrects = 0
        for batch_idx, (images, _, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(output))
            corrects += (preds == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * corrects / len(train_loader.dataset)
        return avg_loss, accuracy
                
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        corrects = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, _, labels in val_loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                
                output = self.model(images)
                loss = self.loss_fn(output, labels)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(output))
                corrects += (preds == labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())


        avg_loss = val_loss / len(val_loader.dataset)
        accuracy = 100. * corrects / len(val_loader.dataset)
        auc = roc_auc_score(all_targets, all_preds)
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return avg_loss, accuracy, auc, sensitivity, specificity
    
    
    def run(self, train_loader, val_loader, test_loader, epochs):
        global global_best_val_acc, global_best_param, f_t_acc, f_t_auc, f_t_sen, f_t_spe
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train(train_loader)
            val_loss, val_acc, auc, sen, spe = self.validate(val_loader)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                ##torch.save(self.model.state_dict(), 'best_model.pth')
                self.best_param = pickle.dumps(self.model.state_dict()) 
                self.epochs_without_improvement = 0

                print(f"Epoch {epoch+1}/{epochs}")
                print("-"*15)
                print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
                print(f"val Acc: {val_acc:.2f}% val AUC: {auc:.2f} val Sensitivity: {sen:.2f} val Specificity: {spe:.2f}")

                _, t_acc, t_auc, t_sen, t_spe = self.validate(test_loader)
                print(f"test Acc: {t_acc:.2f}% test AUC: {t_auc:.2f} test Sensitivity: {t_sen:.2f} test Specificity: {t_spe:.2f}")
                
                if val_acc > global_best_val_acc:
                    global_best_val_acc = val_acc
                    global_best_param = pickle.dumps(self.model.state_dict())
                    f_t_acc, f_t_auc, f_t_sen, f_t_spe = t_acc, t_auc, t_sen, t_spe

            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement > self.patience:
                    print("Early stopping")
                    break

            self.scheduler.step(val_acc)
        ##self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.load_state_dict(pickle.loads(self.best_param))




#global global_best_val_acc, global_best_param, f_t_acc, f_t_auc, f_t_sen, f_t_spe
global_best_val_acc = float(0)
class Trainer_geometry_cnn:
    def __init__(self, model, device, patience=15, lr = 0.0001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4) # optim.Adadelta(model.parameters(), lr=1.0) #
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.7, patience=patience//2, verbose=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.patience = patience
        self.best_val_acc = float(0)
        self.best_param = pickle.dumps(self.model.state_dict())
        self.epochs_without_improvement = 0
        
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        corrects = 0
        for batch_idx, (images, landmarks, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(images, landmarks)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(output))
            corrects += (preds == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * corrects / len(train_loader.dataset)
        return avg_loss, accuracy
                
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        corrects = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, landmarks, labels in val_loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                
                output = self.model(images, landmarks)
                loss = self.loss_fn(output, labels)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(output))
                corrects += (preds == labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())


        avg_loss = val_loss / len(val_loader.dataset)
        accuracy = 100. * corrects / len(val_loader.dataset)
        auc = roc_auc_score(all_targets, all_preds)
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return avg_loss, accuracy, auc, sensitivity, specificity
    
    
    def run(self, train_loader, val_loader, test_loader, epochs):
        global global_best_val_acc, global_best_param, f_t_acc, f_t_auc, f_t_sen, f_t_spe
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train(train_loader)
            val_loss, val_acc, auc, sen, spe = self.validate(val_loader)

            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                ##torch.save(self.model.state_dict(), 'best_model.pth')
                self.best_param = pickle.dumps(self.model.state_dict()) 
                self.epochs_without_improvement = 0
                
                print(f"Epoch {epoch+1}/{epochs}")
                print("-"*15)
                print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
                print(f"val Acc: {val_acc:.2f}% val AUC: {auc:.2f} val Sensitivity: {sen:.2f} val Specificity: {spe:.2f}")
                
                _, t_acc, t_auc, t_sen, t_spe = self.validate(test_loader)
                print(f"test Acc: {t_acc:.2f}% test AUC: {t_auc:.2f} test Sensitivity: {t_sen:.2f} test Specificity: {t_spe:.2f}")
                
                if val_acc > global_best_val_acc:
                    global_best_val_acc = val_acc
                    global_best_param = pickle.dumps(self.model.state_dict())
                    f_t_acc, f_t_auc, f_t_sen, f_t_spe = t_acc, t_auc, t_sen, t_spe
                
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement > self.patience:
                    print("Early stopping")
                    break
            
            self.scheduler.step(val_acc)
        ##self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.load_state_dict(pickle.loads(self.best_param))
        

torch.manual_seed(SEED)

learning_rate = 0.0001
cnn_lstm = CNN_LSTM()
fgpm = FGPM()
###trainer = Trainer(cnn_lstm, device, lr = learning_rate)
###trainer.run(train_loader, val_loader, test_loader, epochs = 15)
geometry_cnn = DeepFakeDetector(cnn_lstm, fgpm)
trainer = Trainer_geometry_cnn(geometry_cnn, device, lr = learning_rate)
trainer.run(train_loader, val_loader, test_loader, epochs = 15)


# Testing Evaluation of the best model in the validation set
print('\n\n------Testing Results of the best model identified for the Validation set------\n')
print(f"test Acc: {f_t_acc:.2f}% \ntest AUC: {f_t_auc:.3f} \ntest Sensitivity: {f_t_sen:.3f} \ntest Specificity: {f_t_spe:.3f}\n\n")





# # ------------------- code for landmarks extraction ----------------------

# # Load the detector
# detector = dlib.get_frontal_face_detector()

# # Load the predictor
# predictor = dlib.shape_predictor("/kaggle/input/facial-predictor/shape_predictor_68_face_landmarks.dat")

# def extract_regions_landmarks(landmarks, start, end):
#     """Extract a facial feature from an image using its landmarks."""
#     points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(start, end+1)])
        
#     return points

# #     Face boundary: 0 to 16
# #     Left eyebrow: 17 to 21
# #     Right eyebrow: 22 to 26
# #     Left eye: 36 to 41
# #     Right eye: 42 to 47
# #     Nose: 27 to 35
# #     Mouth: 48 to 67

# fake_dir = "/kaggle/input/real-and-fake-face-detection/real_and_fake_face/training_fake/"
# real_dir = "/kaggle/input/real-and-fake-face-detection/real_and_fake_face/training_real/"
# fake_images_paths = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
# real_images_paths = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
# all_images_paths = fake_images_paths + real_images_paths

# #cnt = 0
# for img_path in all_images_paths:
#     #cnt+=1
#     #if cnt==20: break
#     img = Image.open(img_path)
#     img = np.array(img)
    
#     # ----- HERE is the def ----
#     #img = img*255
#     img = img.astype('uint8')

#     # Detect faces
#     faces = detector(img)
    
#     if len(faces) == 0:
#             ###print('No Face Found')
#             region_landmarks = fill_no_face_cases()
            
#     for face in faces:
        
#         x1 = face.left()
#         y1 = face.top()
#         x2 = face.right()
#         y2 = face.bottom()
#         # Extract face
#         extracted_face = img[y1:y2, x1:x2]
#         if extracted_face.size == 0:
#             ##print('No Face Found') maybe for some very few instances, face could not be found, if for example is turned position or very blurred
#             region_landmarks = fill_no_face_cases()
#         else:

#     #         plt.figure()
#     #         plt.imshow(extracted_face)
#     #         plt.show()

#             # Get landmarks
#             landmarks = predictor(image=img, box=face)

#             # Extract facial landmarks
#             left_eye = extract_regions_landmarks(landmarks, 36, 41)
#             right_eye = extract_regions_landmarks(landmarks, 42, 47)
#             nose = extract_regions_landmarks(landmarks, 27, 35)
#             mouth = extract_regions_landmarks(landmarks, 48, 67)
#             left_eyebrow = extract_regions_landmarks(landmarks, 17, 21)
#             right_eyebrow = extract_regions_landmarks(landmarks, 22, 26)

#             region_landmarks = {"LEy":left_eye,
#                                "LEb":left_eyebrow,
#                                "REy":right_eye,
#                                "REb":right_eyebrow,
#                                "N":nose,
#                                "M":mouth}
#             break

#     save_data(img_path, img, region_landmarks)

    
# # # so all the landmark pairs will be:
# # LEy - LEb, LEy - REy, LEy - REb, LEy - N, LEy - M, 
# #            LEb - REy, LEb - REb, LEb - N, LEb - M, 
# #                       REy - REb, REy - N, REy - M, 
# #                                  REb - N, REb - M, 
# #                                           N - M
# # # This is required for Computing distances between pairs of landmarks

# # # This is required for Computing angles between sets of three landmarks

# # # Compute areas of regions defined by sets of landmarks
    
