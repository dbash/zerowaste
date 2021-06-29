# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Randomized Image Sampling for Explanations (RISE)

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import torchvision.models as models

from utils import *
from explanations import *


# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import torchvision as tvis
import time
from utils import *
cudnn.benchmark = True


# %%
### Load the dataset


# %%
input_size = (224, 224)
batch_size = 16
feature_extract=True


# %%
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


# %%
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



# %%
img, labels = next(iter(validation_loader))
print()



# %%
model = models.resnet50(True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = nn.Sequential(model, nn.Softmax())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= nn.DataParallel(model)
model = model.to(device)


# %%
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


# %%
criterion = torch.nn.BCELoss()
dataloaders_dict = {"train": train_loader, "val": validation_loader}


# %%
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


# %%
train_from_scratch = False
ckpt_path = "binary_classifier.pt"

if train_from_scratch:
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=4)
    torch.save(model_ft.state_dict(), ckpt_path)
else:
    model.load_state_dict(torch.load(ckpt_path))


# %%
for p in model.parameters():
    p.requires_grad = False
model.eval()


# %%

for img, labels in dataloaders_dict["val"]:
    img = img.to(device)
    labels = labels.to(device)
    outputs = model(img)
    preds = (outputs > 0.5).type(torch.cuda.FloatTensor)
    print(preds[:5])
    print(labels.data[0:5])
    break

# %% [markdown]
# ---
# %% [markdown]
# ## RISE

# %%
#model = nn.DataParallel(model)


# %%
explainer = RISE(model, input_size, batch_size)


# %%
"""
maskspath = 'masks.npy'
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=3000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')
"""

# %% [markdown]
# ---
# %% [markdown]
# ## Running explanations

# %%
def class_name(idx):
    return "after" if idx == 0 else "before"


# %%
import PIL
from PIL import Image

def load_img(path):
    img = Image.open(path).resize(input_size, PIL.Image.BILINEAR)
    x = np.asarray(img)
    x = np.expand_dims(x, axis=0)
    return x


# %%
def example(img, top_k=2):
    saliency = explainer(img.cuda()).cpu().numpy()
    p, c = torch.topk(model(img.cuda()), k=2)
    p, c = p[0], c[0]
    
   
    plt.figure(figsize=(10, 5*top_k))
    for k in range(top_k):
        plt.subplot(top_k, 2, 2*k+1)
        plt.axis('off')
        plt.title('%.2f %s' % (100*float(p[k]),
                                     class_name(int(c[k]))))
        tensor_imshow(img[0])

        plt.subplot(top_k, 2, 2*k+2)
        plt.axis('off')
        plt.title(class_name(int(c[k])))
        tensor_imshow(img[0])
        sal = saliency[c[k]]
        plt.imshow(sal, cmap='jet', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()


# %%
img_file = "/home/guest/mabdelfa/zerowaste/data/zerowaste-w/before/01_frame_004640.PNG"
pred = (model(read_tensor(img_file).cuda()) > 0.5)[0]

print(pred)


# %%
#example(read_tensor(img_file), 2)


# %%
def explain_all(data_loader, explainer):
    # Get all predicted labels first
    target = np.empty(len(data_loader), np.int)
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
        p, c = torch.max(model(img.cuda()), dim=1)
        target[i] = c[0]

    # Get saliency maps for all images in val loader
    explanations = np.empty((len(data_loader), *input_size))
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
        saliency_maps = explainer(img.cuda())
        explanations[i] = saliency_maps[target[i]].cpu().numpy()
    return explanations


# %%
rangee = range(95, 105)
n_batch = 1

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=n_batch, shuffle=False,
    num_workers=8, pin_memory=True, sampler=RangeSampler(rangee))

#explanations = explain_all(data_loader, explainer)


# %%
def explain_all_batch(data_loader, explainer):
    n_batch = len(data_loader)
    b_size = data_loader.batch_size
    total = n_batch * b_size
    # Get all predicted labels first
    target = np.empty(total, 'int64')
    for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
        p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
        target[i * b_size:(i + 1) * b_size] = c.cpu()
    image_size = imgs.shape[-2:]
    count = 0

    # Get saliency maps for all images in val loader
    explanations = np.empty((total, *image_size))
    for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
        saliency_maps = explainer(imgs.cuda())
        explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
            range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
    return explanations


# %%
explainer = RISEBatch(model, input_size, batch_size)


# %%
maskspath = 'masks_for_batch.npy'
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=3000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')


# %%
imgs_per_batch = 2

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=imgs_per_batch, shuffle=False,
    num_workers=8, pin_memory=True)

explanations = explain_all_batch(data_loader, explainer)

explanations.tofile('my_explanations.npy')