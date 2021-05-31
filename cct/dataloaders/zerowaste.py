from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json

def text_to_list(text_path):
    txt_file = open(text_path, "r")
    content = txt_file.read()
    content_list = content.split("\n")
    content_list = list(set(content_list) - set(""))
    txt_file.close()
    return content_list

class ZeroWasteDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 5

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(ZeroWasteDataset, self).__init__(**kwargs)

    def _set_files(self):
        #self.root = os.path.join(self.root, 'data')
        if self.split == "val" or self.split == "test":
            file_list = os.path.join("dataloaders/zerowaste_splits", f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join("dataloaders/zerowaste_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        self.files = text_to_list(file_list)
        #print(self.files)
        # self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, "data", self.files[index])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_name = os.path.basename(self.files[index])#.split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_name)
        elif "unsupervised" in self.split:
            label = np.zeros((image.shape[0], image.shape[1]))
            return image, label, image_name
        else:
            label_path = os.path.join(self.root, "sem_seg", self.files[index])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label, image_name

class ZeroWaste(BaseDataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = ZeroWasteDataset(**kwargs)

        super(ZeroWaste, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
