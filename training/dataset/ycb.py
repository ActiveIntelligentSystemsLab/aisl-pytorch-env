import torch
import os
import cv2
from PIL import Image
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class SyntheticClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root, size=224, is_training=True):
        # let us assume that you have data directory set-up like this
        # + vision_datasets
        # +++++ train.txt // each line in this file contains mapping about RGB image and class <image1.jpg>, Class_id
        # +++++ val.txt // each line in this file contains mapping about RGB image and class <image1.jpg>, Class_id
        # +++++ images // This directory contains RGB images
        # +++++++++ image1.jpg
        # +++++++++ image2.jpg

        self.train = is_training
        if self.train:
            data_file = os.path.join(root, 'train.lst')
        else:
            data_file = os.path.join(root, 'val.lst')

        self.images = []
        self.labels = []
        with open(data_file, 'r') as lines:
            for line in lines:
                # line is a comma separated file that contains mapping between RGB image and class iD
                # <image1.jpg>, Class_ID
                line_split = line.split(',') # index 0 contains rgb location and index 1 contains label id
                rgb_img_loc = line_split[0].strip()
                label_id = int(line_split[1].strip()) - 1#strip to remove spaces
                assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)
                self.labels.append(label_id)

        self.transform = self.transforms(size=size, is_training=is_training)

    def transforms(self, size=None, is_training=True):
        if is_training:
            return A.Compose(
            [
#                A.SmallestMaxSize(max_size=160),
#                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#                A.RandomCrop(height=128, width=128),
                A.Resize(height=size, width=size),
                A.GaussNoise(p=0.7),
                A.GaussianBlur(p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.CenterCrop(height=128, width=128),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def generate_background(self, obj):
        """ Generate a background that matches the size of an object image
        
        Args:
            obj (OpenCV): An object image.

	    Returns:
	        OpenCV image: Generated background image
        """
        # Take max of the height and the width of the object image
        size = max(obj.shape[:2])
        bg = np.ones((size, size, 3), dtype=np.uint8)

        # Set random value to each color channel
        bg[:,:,0] *= random.randint(0, 255)
        bg[:,:,1] *= random.randint(0, 255)
        bg[:,:,2] *= random.randint(0, 255)
        
        return bg
    
    def synthesize_image(self, obj, background):
        """ Generate a training image by blending an object image and a background image
        
        Args:
            obj (OpenCV): An object image.
            background (OpenCV): A background image. Width and height have to be bigger than those of 'obj'

        Returns:
	        OpenCV image: Synthesized training image
        """
        # Get the sizes of the images
        h_b = background.shape[0]
        w_b = background.shape[1]
        h_o = obj.shape[0]
        w_o = obj.shape[1]
        
        # Calculate the location 
        h_start = h_b // 2 - h_o // 2
        h_end   = h_start + h_o
        w_start = w_b // 2 - w_o // 2
        w_end   = w_start + w_o
        
        # If there is the alpha channel, create mask
        if len(obj.shape) == 3 and obj.shape[2] == 4:
            # True indicates non-zero alpha
            mask = obj[:,:,3] != 0
            # Copy values of pixels whose alpha is non-zero
            background[h_start:h_end,w_start:w_end][mask] = obj[mask][:,:3]
        else:
            background[h_start:h_end,w_start:w_end] = obj

        return background

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        obj_img = cv2.imread(self.images[index], -1)
        label_id = self.labels[index]

        bg = self.generate_background(obj_img)
        rgb_img = self.synthesize_image(obj_img, bg)
        rgb_img = self.transform(image=rgb_img)

        return rgb_img, label_id