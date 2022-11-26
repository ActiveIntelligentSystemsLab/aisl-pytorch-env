# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

# ROS related
import rospy
from sensor_msgs.msg import Image

# PyTorch related
import torch
from torchvision import transforms
# from util.util import import_model


class InferenceNodeBase(object):
    def __init__(self, model_name: str, version: str, import_model_func):
        self.model_name = model_name
        self.version = version

        # Import model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rospy.loginfo(self.device)
        if model_name:
            try:
                model_etc = import_model_func(model_name, version)
                self.model = model_etc["model"]

                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
                    ]) if model_etc["transform"] is None else model_etc["transform"]
            except ValueError:
                print("Invalid input ('{}', '{}')".format(model_name, version))
                exit(1)

            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None

        self.image_sub = rospy.Subscriber(
            'image', Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher('visualize', Image, queue_size=10)

    def image_callback(self, img_msg):
        """ Take an image message and process it through the model

        Args:
            img_msg: Image message
        """
        raise NotImplementedError
