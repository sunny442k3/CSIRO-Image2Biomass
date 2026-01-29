import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(IMG_SIZE=(512, 512)):
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=(-5, 5), p=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ], p=1.0)

def get_valid_transforms(IMG_SIZE=(512, 512)):
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ], p=1.0)
