from .transforms import transforms_refine as transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from .imutils import RandomResizeLong, RandomCrop
from PIL import Image
import os, os.path
from torch.utils.data.sampler import SubsetRandomSampler
#from .vision import VisionDataset
from torchvision.datasets.vision import VisionDataset

import torchvision.datasets as datasets
from PIL import Image


from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:

    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances



class DatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            directory_root: str,
            mode: str,
            crop_size,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.directory_root = directory_root
        self.org_root = self.directory_root + '/org'
            

        classes, class_to_idx = self.find_classes(self.root)
        classes_org, class_to_idx_org = self.find_classes(self.org_root)

        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        samples_org = self.make_dataset(self.org_root, class_to_idx_org, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.samples_org = samples_org
        self.targets = [s[1] for s in samples]

        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)



    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.samples[index]
        sample = Image.open(path)

        path_org, target_org = self.samples_org[index]
        sample_org = self.loader(path_org)
        
        #print(os.path.splitext(path_leaf(path))[0])

        att_map_path = self.directory_root + '/localization_maps/' + os.path.splitext(path_leaf(path))[0] + '.npy'
        att_map = np.load(att_map_path)

        if self.transform is not None:
            sample_org, sample, sample, att_map = self.transform(sample_org, sample, sample, att_map)

        if self.mode != 'test':
            maximum_mask = (att_map == att_map.max(0, keepdim=True)[0]).float()
            att_map = att_map * maximum_mask

        if self.target_transform is not None:
            target = self.target_transform(target)
            target_org = self.target_transform(target_org)

        return sample_org, target, sample, att_map, path

        """ img_name =  self.image_list[idx]
        label = self.label_list[idx]
        
        image = Image.open(img_name).convert('RGB')
        sal_map = Image.open(self.sal_map_list[idx])
        att_map = np.load(self.att_map_list[idx])
        
        if self.transform is not None:
            image, sal_map, gt_map, att_map = self.transform(image, sal_map, gt_map, att_map)

        if self.mode != 'test':
            maximum_mask = (att_map == att_map.max(0, keepdim=True)[0]).float()
            att_map = att_map * maximum_mask
            
        return image, label, sal_map, gt_map, att_map, img_name """


    def __len__(self) -> int:
        return len(self.samples)



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ZeroWasteDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            directory_root: str,
            mode: str,
            crop_size,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ZeroWasteDataset, self).__init__(root, directory_root, mode, crop_size, loader,IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

def ZeroWaste_data_loaders(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
        
    

    target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x, dtype=torch.int64), 2)

    d_root = args.img_dir
    org_root = args.img_dir + "/seg/val_test"
    tsfm_train = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.RandomCrop(crop_size),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])
    dataset = ZeroWasteDataset(root = org_root,directory_root=d_root, mode='train',crop_size=crop_size,transform=tsfm_train,
                              target_transform=target_transform) 
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    seg_root_val = args.img_dir + "/seg/val_test"
    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])
    dataset = ZeroWasteDataset(root = seg_root_val,directory_root=d_root, mode='valid',crop_size=crop_size,transform=tsfm_test,
                              target_transform=target_transform)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    seg_root_test = args.img_dir + "/seg/test"
    dataset = ZeroWasteDataset(root = seg_root_test,directory_root=d_root, mode='test',crop_size=crop_size,transform=tsfm_test,
                              target_transform=target_transform)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


def train_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
        
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.RandomCrop(crop_size),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset(args.train_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, mode='train')

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_loader


def valid_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
        
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDataset(args.test_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, mode='valid')

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader



def test_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
        
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(crop_size, keep=True),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDataset(args.test_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, mode='test')

    test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    return test_loader


class VOCDataset(Dataset):
    def __init__(self, datalist_file, input_size, root_dir, num_classes=20, transform=None, mode='train'):
        self.root_dir = root_dir
        self.input_size = input_size
        self.mode = mode
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes

        self.image_list, self.label_list, self.gt_map_list, self.sal_map_list, self.att_map_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        
        
    def __len__(self):
        return len(self.image_list)

    
    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        label = self.label_list[idx]
        
        image = Image.open(img_name).convert('RGB')
        sal_map = Image.open(self.sal_map_list[idx])
        gt_map = Image.open(self.gt_map_list[idx])
        att_map = np.load(self.att_map_list[idx])
        
        if self.transform is not None:
            image, sal_map, gt_map, att_map = self.transform(image, sal_map, gt_map, att_map)

        if self.mode != 'test':
            maximum_mask = (att_map == att_map.max(0, keepdim=True)[0]).float()
            att_map = att_map * maximum_mask
            
        return image, label, sal_map, gt_map, att_map, img_name
        
    
    def read_labeled_image_list(self, data_dir, data_list):
        img_dir = os.path.join(data_dir, "JPEGImages")
        gt_map_dir = os.path.join(data_dir, "SegmentationClassAug/")
        sal_map_dir = os.path.join(data_dir, "saliency_map/")
        att_map_dir = os.path.join(data_dir, "localization_maps")
        
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        gt_map_list = []
        sal_map_list = []
        att_map_list = []
        
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            gt_map = fields[0] + '.png'
            sal_map = fields[0] + '.png'
            att_map = fields[0] + '.npy'
            
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(img_dir, image))
            gt_map_list.append(os.path.join(gt_map_dir, gt_map))
            sal_map_list.append(os.path.join(sal_map_dir, sal_map))
            att_map_list.append(os.path.join(att_map_dir, att_map))
            img_labels.append(labels)
            
        return img_name_list, img_labels, gt_map_list, sal_map_list, att_map_list
    