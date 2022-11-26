#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

# ROS related
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# PyTorch related
import torch
from torchvision import transforms
from util.util import import_model

# Other
import json

from rospy_util.utils import imgmsg_to_pil, pil_to_imgmsg, write_on_pil
from rospy_util.inference_node_base import InferenceNodeBase


# class ObjectRecognitionNode(object):
class ObjectRecognitionNode(InferenceNodeBase):
    def __init__(self, model_name: str, version: str, label_map_path: str):
        super().__init__(model_name, version, import_model)

        # Load ImageNet class names
        self.labels_map = json.load(open(label_map_path))
        self.labels_map = [self.labels_map[str(i)] for i in range(1000)]

        # Additional publisher
        self.result_pub = rospy.Publisher('top_label', String, queue_size=10)
        self.bridge = CvBridge()

    def image_callback(self, img_msg):
        """ Take an image message and process it through the model

        Args:
            img_msg: Image message
        """
        pil_image, _, _ = imgmsg_to_pil(img_msg)
        tensor_image = self.transforms(pil_image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor_image.to(self.device))

        is_top = True
        for idx in torch.topk(output, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(output, dim=1)[0, idx].item()
            res_str = '{label:<75} ({p:.2f}%)'.format(
                label=self.labels_map[idx], p=prob*100)
            if is_top:
                rospy.loginfo(res_str)

                write_on_pil(pil_image, res_str)
                img_msg_pub = pil_to_imgmsg(pil_image)

                # Publish image for visualization and the label of the top object
                self.image_pub.publish(img_msg_pub)
                self.result_pub.publish(res_str)
                is_top = False


def main():
    """Main function to initialize the ROS node"""
    rospy.init_node("object_recognition_node")

    # Get ROS params
    model_name = rospy.get_param('~model_name', 'resnet')
    version = rospy.get_param('~version', '18')
    label_map_path = rospy.get_param('~label_map_path', '')

    obj_recog_node = ObjectRecognitionNode(model_name, version, label_map_path)

    rospy.loginfo('object_recognition_node is initialized')

    rospy.spin()


if __name__ == '__main__':
    main()
