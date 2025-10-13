import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import transforms
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation
from semseg.augmentations_mm import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomScale, RandomCrop
from PIL import Image, ImageFilter

class MFNet(Dataset):
    """
    num_classes: 40
    """
    CLASSES = ['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
    # CLASSES = ['E','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    # 'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    # 'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    # 'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

    PALETTE = torch.tensor([[64,0,128],[64,64,0],[0,128,192],[0,0,192],[128,128,0],[64,64,128],[192,128,128],[192,64,0]])

    def __init__(self, root: str = 'data/MFNet', split: str = 'train', transform = None, modals = ['img', 'depth'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']

        # pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Normalize([0.449], [0.226]),
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.split = split
        self.transform = transform
        # scale_range = tuple(float(i) for i in "0.5 2.0".split(' '))
        # crop_size = tuple(int(i) for i in "480 640".split(' '))

        

        self.class_weight = np.array([4.01302219, 5.17995767, 12.47921102, 13.79726557, 18.47574439, 19.97749822,
                                      21.10995738, 25.86733191, 27.50483598, 27.35425244, 25.12185149, 27.04617447,
                                      30.0332327, 29.30994935, 34.72009825, 33.66136128, 34.28715586, 32.69376342,
                                      33.71574286, 37.0865665, 39.70731054, 38.60681717, 36.37894266, 40.12142316,
                                      39.71753044, 39.27177794, 43.44761984, 42.96761184, 43.98874667, 43.43148409,
                                      43.29897719, 45.88895515, 44.31838311, 44.18898992, 42.93723439, 44.61617778,
                                      47.12778303, 46.21331253, 27.69259756, 25.89111664, 15.65148615, ])

        self.root = root
        # self.transform = transform
        self.n_classes = 9
        self.ignore_label = 255
        self.modals = modals
        self.files = self._get_file_names(split)
        # self.val_size = Resize(crop_size)
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        # print("item_name", item_name)
        # item_name=item_name.split('/')[1].split('.jpg')[0]
        item_name=item_name
        # print("item_name", item_name)
        rgb = os.path.join(*[self.root, 'RGB', item_name+'.png'])
        x1 = os.path.join(*[self.root, 'Modal', item_name+'.png'])
        lbl_path = os.path.join(*[self.root, 'Label', item_name+'.png'])
        # breakpoint()
        import imageio

        sample = {}
        # image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        # depth = Image.open(os.path.join(self.root, depth_path)).convert('RGB')  # 1 channel -> 3
        # # depth = Image.open(os.path.join(self.root, hha_path))
        # label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37

        sample['image'] = Image.open(rgb)
        if 'depth' in self.modals:
            sample['depth'] = Image.open(x1).convert('RGB')
        if 'lidar' in self.modals:
            raise NotImplementedError()
        if 'event' in self.modals:
            raise NotImplementedError()
        label = Image.open(lbl_path)
        # label[label==255] = 0
        # label -= 1
        sample['label'] = label

        

        if self.transform:
            sample = self.transform(sample)
        # else:
        #     sample = self.val_size(sample)
        
        self.class_weight = np.array([3.3855, 5.6249, 12.0095, 13.9648, 21.0021, 20.5423, 25.7143, 27.3796,
                                      28.7891, 27.1876, 28.5285, 28.1800, 32.1900, 31.0330, 35.3340, 32.6370,
                                      36.1279, 36.2773, 37.1518, 35.4062, 40.4284, 37.5624, 33.7325, 40.8962,
                                      40.9146, 40.2932, 43.6822, 44.7070, 43.7316, 42.5433, 45.0084, 44.8754,
                                      45.2884, 44.5937, 45.4827, 45.2658, 46.5859, 46.4504, 26.3095, 27.8991,
                                      16.6582])

        # label = sample['mask']
        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        # del sample['mask']
        # label = self.encode(label.squeeze().numpy()).long()
        # sample = [sample[k] for k in self.modals]
        return sample

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(self.root, 'test.txt') if split_name == 'val' else os.path.join(self.root, 'train2.txt')
        file_names = []
        with open(source) as f:
            files = f.readlines()
        for item in files:
            file_name = item.strip()
            if ' ' in file_name:
                file_name = file_name.split(' ')[0]
            file_names.append(file_name)
        return file_names
    
    @property
    def cmap(self):
        return [(0, 0, 0),
                (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0),
                (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128),
                (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
                (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
                (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64),
                (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192),
                (0, 128, 192), (128, 128, 192), (64, 0, 64)]


if __name__ == '__main__':
    traintransform = get_train_augmentation((480, 640), seg_fill=255)

    trainset = NYU(transform=traintransform, split='val')
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))
