import torch
import os
import cv2
from PIL import Image
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize
from torchvision.transforms import functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import copy

ycb_class_list = [
    "001_chips_can",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "020",
    "021_bleach_cleanser",
    "022_windex_bottle",
    "023_wine_glass",
    "024_bowl",
    "025",
    "026_sponge",
    "027",
    "028",
    "029_plate",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "034",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "038_padlock",
    "039_key",
    "040_large_marker",
    "041_small_marker",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "045",
    "046",
    "047_plastic_nut",
    "048_hammer",
    "049_small_clamp",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "059_chain",
    "060",
    "061_foam_brick",
    "062_dice",
    "063-marbles",
    "064",
    "065-cups",
    "066",
    "067",
    "068",
    "069",
    "070-a_colored_wood_blocks",
    "071_nine_hole_peg_test",
    "072-toy_airplane",
    "073-lego_duplo",
    "074",
    "075",
    "076_timer",
    "077_rubiks_cube"
]

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
            ])
        else:
            return A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.CenterCrop(height=128, width=128),
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
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = self.transform(image=rgb_img)["image"]
        orig_img = copy.deepcopy(rgb_img)

        rgb_img = F.to_tensor(Image.fromarray(rgb_img))
        rgb_img = F.normalize(rgb_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalize the tensor
        orig_img = F.to_tensor(Image.fromarray(orig_img))

        return {"rgb_img": rgb_img, "orig_img": orig_img, "label_id": label_id}