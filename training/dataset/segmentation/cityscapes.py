from torchvision.datasets import Cityscapes
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
from cityscapesscripts.helpers.labels import labels as label_table

class_wts_list = [
    10 / 2.8149201869965,
    10 / 6.9850029945374,
    10 / 3.7890393733978,
    10 / 9.9428062438965,
    10 / 9.7702074050903,
    10 / 9.5110931396484,
    10 / 10.311357498169,
    10 / 10.026463508606,
    10 / 4.6323022842407,
    10 / 9.5608062744141,
    10 / 7.8698215484619,
    10 / 9.5168733596802,
    10 / 10.373730659485,
    10 / 6.6616044044495,
    10 / 10.260489463806,
    10 / 10.287888526917,
    10 / 10.289801597595,
    10 / 10.405355453491,
    10 / 10.138095855713,
]

color_encoding = OrderedDict([
    ('road', (128, 64, 128)),
    ('sidewalk', (244, 35, 232)),
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)),
    ('fence', (190, 153, 153)),
    ('pole', (153, 153, 153)),
    ('traffic light', (250, 170, 30)),
    ('traffic sign', (220, 220,  0)),
    ('vegetation', (107, 142, 35)),
    ('terrain', (152, 251, 152)),
    ('sky', (70, 130, 180)),
    ('person', (220, 20, 60)),
    ('rider', (255,  0,  0)),
    ('car', (0,  0, 142)),
    ('truck', (0,  0, 70)),
    ('bus', (0, 60, 100)),
    ('train', (0, 80, 100)),
    ('motorcycle', (0,  0, 230)),
    ('bicycle', (119, 11, 32)),
    ('background', (0,  0,  0)),
])

trainIdList = np.array([i.trainId for i in label_table])


class AlbCityscapes(Cityscapes):
    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "instance",
        transforms: Optional[Callable] = None,
        alb_transforms: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, split=split, mode=mode, target_type=target_type,
                         transforms=transforms)

        # Albumentation.Compose
        self.alb_transforms = alb_transforms

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        image = np.array(image)
        # Convert ID to trainID
        target = np.array(target, dtype=np.uint64)
        target[target == 255] = 0  # Remove '-1'
        target = trainIdList[target]  # Convert ID to trainId

        # Data augmentation
        if self.alb_transforms is not None:
            transformed = self.alb_transforms(image=image, mask=target)
            image, target = transformed['image'], transformed['mask']
        elif self.transforms is not None:
            image = Image.fromarray(image)
            target = Image.fromarray(target).convert('L')
            image, target = self.transforms(image, target)

        # Normalize and convert to tensor
        transform_train = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        image_orig = ToTensorV2()(image=image)['image']
        image = transform_train(image=image)['image']

        return {"rgb_img": image, "label": target, "orig_img": image_orig}
