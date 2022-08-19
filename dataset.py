import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, type, size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        
        if type == "train":
            self.transform = A.Compose([
                A.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5), # Data augmentation
                A.RandomCrop(size, size),
                A.Normalize(
                    mean=[0, 0, 0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2()
            ])
        elif type == "validation":
            self.transform = A.Compose([
                A.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_NEAREST),
                A.CenterCrop(size, size),
                A.Normalize(
                    mean=[0, 0, 0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2()
            ])
        else:
            raise Exception("Invalid dataset type")

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[index]).convert("L"), dtype=np.float32)

        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            # print(self.image_paths[index])
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask

def get_loaders(img_dir, mask_dir, batch_size, n_workers, pin_memory):

    image_paths = glob.glob(img_dir)
    mask_paths = glob.glob(mask_dir)

    train_image_paths = image_paths[:int(0.9 * len(image_paths))]
    train_mask_paths = mask_paths[:int(0.9 * len(mask_paths))]
    val_image_paths = image_paths[int(0.9 * len(image_paths)):]
    val_mask_paths = mask_paths[int(0.9 * len(mask_paths)):]

    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, "train", 256)
    val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, "validation", 256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory)

    return train_loader, val_loader
