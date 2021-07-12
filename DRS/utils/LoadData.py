from .transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from .imutils import RandomResizeLong, RandomCrop
import os
from PIL import Image
import random
import os, os.path
from torch.utils.data.sampler import SubsetRandomSampler
#from .vision import VisionDataset
from torchvision.datasets.vision import VisionDataset

import torchvision.datasets as datasets
from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
#from utilss import *


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
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
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
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            root_org: str,
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
        self.root_org = root_org
        classes, class_to_idx = self.find_classes(self.root)
        classes_org, class_to_idx_org = self.find_classes(self.root_org)

        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        samples_org = self.make_dataset(self.root_org, class_to_idx_org, extensions, is_valid_file)

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

        if self.mode == 'valid':
            self.map_transform = transforms.Compose([transforms.Resize(crop_size, Image.NEAREST), transforms.From_Numpy()])
        elif self.mode == 'test':
            self.map_transform = transforms.Compose([transforms.From_Numpy(), ])  # pseudo gt gen


    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)



    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        #sample = self.loader(path)
        sample = Image.open(path)

        path_org, target_org = self.samples_org[index]
        sample_org = self.loader(path_org)

        if self.transform is not None:
            #sample = self.transform(sample)
            sample_org = self.transform(sample_org)
        if self.target_transform is not None:
            target = self.target_transform(target)
            target_org = self.target_transform(target_org)

            if (target != target_org):
                print('targets are not equal!!!!!!!!!!!!!!!!!!!')
                quit()
        
        sample = self.map_transform(sample)
        
        sample = (sample > 50)
            
        
        

        return sample_org, target, sample


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
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            root_org: str,
            mode: str,
            crop_size,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ZeroWasteDataset, self).__init__(root, root_org, mode, crop_size, loader,IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

def ZeroWaste_data_loaders(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
        
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.RandomCrop(crop_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    zz_input_size = (224, 224)
    #batch_size = 16
    #feature_extract=True

    """ transform = {
        'train': transforms.Compose(
            [   
                transforms.Resize([230, 230]),
                transforms.RandomAffine(degrees=[-60, 60], ),
                transforms.RandomCrop(zz_input_size[0]),
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
            [transforms.Resize(zz_input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        } """

    target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x, dtype=torch.int64), 2)

    org_root = args.img_dir + "/org"
    seg_root = args.img_dir + "/seg"

    dataset = datasets.ImageFolder(root = org_root, transform=tsfm_train,
                                target_transform=target_transform)
    validation_split = .2
    #test_split = .1
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    #val_split_idx = int(np.floor(validation_split * dataset_size))
    #test_split_idx = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    #train_indices, val_indices = indices[val_split_idx:], indices[:val_split_idx]

    #train_indices, val_indices, test_indices = indices[val_split_idx+val_split_idx:], indices[test_split_idx:val_split_idx+test_split_idx], indices[:test_split_idx]
    train_indices = indices
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    #valid_sampler = SubsetRandomSampler(val_indices)
    #test_sampler = SubsetRandomSampler(test_indices)

    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
    #                                        sampler=train_sampler)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                                sampler=valid_sampler)
    #test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                               sampler=test_sampler)

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    #img_test = VOCDataset(args.test_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, mode='valid')

    #val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    dataset = ZeroWasteDataset(root = seg_root,root_org=org_root, mode='valid',crop_size=crop_size,transform=tsfm_test,
                              target_transform=target_transform)

    #dataset_size = len(dataset)
    #val_indices = list(range(dataset_size))
    #valid_sampler = SubsetRandomSampler(val_indices)


    #val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                            sampler=valid_sampler)
    
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dataset = ZeroWasteDataset(root = seg_root,root_org=org_root, mode='test',crop_size=crop_size,transform=tsfm_test,
                              target_transform=target_transform)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader

def train_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
        
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.RandomCrop(crop_size),
                                     transforms.ToTensor(),
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

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return val_loader

def ZeroWaste_valid_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    #img_test = VOCDataset(args.test_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, mode='valid')

    #val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x, dtype=torch.int64), 2)

    org_root = args.img_dir + "/org"
    seg_root = args.img_dir + "/seg"

    dataset = ZeroWasteDataset(root = seg_root,root_org=org_root, transform=tsfm_test,
                              target_transform=target_transform)

    dataset_size = len(dataset)
    val_indices = list(range(dataset_size))
    np.random.shuffle(val_indices)
    valid_sampler = SubsetRandomSampler(val_indices)


    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=valid_sampler)

    return val_loader



def test_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDataset(args.test_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, mode='test')

    test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader


class VOCDataset(Dataset):
    def __init__(self, datalist_file, input_size, root_dir, num_classes=2, transform=None, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes

        self.image_list, self.label_list, self.gt_map_list, self.sal_map_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
            
        if self.mode == 'valid':
            self.map_transform = transforms.Compose([transforms.Resize(input_size, Image.NEAREST), transforms.From_Numpy()])
        elif self.mode == 'test':
            self.map_transform = transforms.Compose([transforms.From_Numpy(), ])  # pseudo gt gen
        
        
    def __len__(self):
        return len(self.image_list)

    
    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.mode != 'train':
            gt_map = Image.open(self.gt_map_list[idx])
            sal_map = Image.open(self.sal_map_list[idx])
            
            gt_map = self.map_transform(gt_map)
            sal_map = self.map_transform(sal_map)
            
            sal_map = (sal_map > 50).float()
            
            return image, self.label_list[idx], sal_map, gt_map, img_name
        
        return image, self.label_list[idx], img_name

    
    def read_labeled_image_list(self, data_dir, data_list):
        img_dir = os.path.join(data_dir, "JPEGImages")
        gt_map_dir = os.path.join(data_dir, "SegmentationClassAug")
        sal_map_dir = os.path.join(data_dir, "saliency_map")
        
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        gt_map_list = []
        sal_map_list = []
        
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            gt_map = fields[0] + '.png'
            sal_map = fields[0] + '.png'
            
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(img_dir, image))
            gt_map_list.append(os.path.join(gt_map_dir, gt_map))
            sal_map_list.append(os.path.join(sal_map_dir, sal_map))
            img_labels.append(labels)
            
        return img_name_list, img_labels, gt_map_list, sal_map_list
