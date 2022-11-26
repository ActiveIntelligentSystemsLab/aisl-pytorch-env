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

        # Get ROS params
        # model_name = rospy.get_param('~model_name', mode_name_default)
        # version = rospy.get_param('~version', version_default)

        # Preprocess image
        self.transforms = transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

        # Import model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rospy.loginfo(self.device)
        try:
            self.model = import_model_func(model_name, version)
        except ValueError:
            print("Invalid input ('{}', '{}')".format(model_name, version))
            exit(1)

        self.model.to(self.device)
        self.model.eval()

        self.image_sub = rospy.Subscriber('image', Image, self.image_callback)
        self.image_pub = rospy.Publisher('visualize', Image, queue_size=10)

    def image_callback(self, img_msg):
        """ Take an image message and process it through the model

        Args:
            img_msg: Image message
        """
        raise NotImplementedError
