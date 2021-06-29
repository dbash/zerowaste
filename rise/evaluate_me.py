import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import torchvision.models as models

from utils import *
from explanations import *

import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision as tvis
import time
cudnn.benchmark = True

from torch.nn.functional import conv2d

from evaluation import CausalMetric, auc, gkern


input_size = (224, 224)
batch_size = 16
feature_extract=True

transform = {
        'train': transforms.Compose(
            [   
                transforms.Resize([230, 230]),
                transforms.RandomAffine(degrees=[-60, 60], ),
                transforms.RandomCrop(input_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, 
                                                   contrast=0.2, 
                                                   saturation=0.2, 
                                                   hue=0.1),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        }

target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x, dtype=torch.int64), 2)

data_folder = "/home/guest/mabdelfa/zerowaste/data/zerowaste-w/"
dataset = datasets.ImageFolder(root = data_folder, transform=transform['train'],
                              target_transform=target_transform)

validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

img, labels = next(iter(validation_loader))

model = models.resnet50(True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = nn.Sequential(model, nn.Softmax())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= nn.DataParallel(model)
model = model.to(device)

params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
optimizer_ft = torch.optim.Adam(params_to_update, lr=0.0005)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_ft, gamma=0.9)

criterion = torch.nn.BCELoss()
dataloaders_dict = {"train": train_loader, "val": validation_loader}

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 1.

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_samples = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.to(torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    labels.data = labels.data
                    loss = criterion(outputs, labels.data)
                    preds = (outputs > 0.5).type(torch.cuda.FloatTensor)
                   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.mean(torch.abs(preds - labels.data))
                num_samples += 1

            epoch_loss = running_loss / batch_size * num_samples
            epoch_acc = running_corrects.double() / num_samples

            print('{} Loss: {:.4f} MSE: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc < best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
            #scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val MSE: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

train_from_scratch = False
ckpt_path = "binary_classifier.pt"

if train_from_scratch:
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=4)
    torch.save(model_ft.state_dict(), ckpt_path)
else:
    model.load_state_dict(torch.load(ckpt_path))

for p in model.parameters():
    p.requires_grad = False
model.eval()


for img, labels in dataloaders_dict["val"]:
    img = img.to(device)
    labels = labels.to(device)
    outputs = model(img)
    preds = (outputs > 0.5).type(torch.cuda.FloatTensor)
    print(preds[:5])
    print(labels.data[0:5])
    break

cudnn.benchmark = True


klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

insertion = CausalMetric(model, 'ins', 224 * 8, substrate_fn=blur)
deletion = CausalMetric(model, 'del', 224 * 8, substrate_fn=torch.zeros_like)

scores = {'del': [], 'ins': []}

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=241, shuffle=False,
    num_workers=8, pin_memory=True, sampler=RangeSampler(range(0, 2410)))

images = np.empty((len(data_loader), 241, 3, 224, 224))
for j, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Loading images')):
    images[j] = img
images = images.reshape((-1, 3, 224, 224))

exp = np.fromfile('my_explanations.npy').reshape((2410, 224, 224))

h = deletion.evaluate(torch.from_numpy(images.astype('float32')), exp, 241)
scores['del'].append(auc(h.mean(1)))

# Evaluate insertion
h = insertion.evaluate(torch.from_numpy(images.astype('float32')), exp, 241)
scores['ins'].append(auc(h.mean(1)))

print('----------------------------------------------------------------')
print('Final:\nDeletion - {:.5f}\nInsertion - {:.5f}'.format(np.mean(scores['del']), np.mean(scores['ins'])))