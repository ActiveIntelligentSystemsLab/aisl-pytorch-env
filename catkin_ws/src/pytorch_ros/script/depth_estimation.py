#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

# ROS related
import rospy
from cv_bridge import CvBridge

# PyTorch related
import torch

from util.util import import_depth_estimation_model
from rospy_util.inference_node_base import InferenceNodeBase


class DepthEstimationNode(InferenceNodeBase):
    def __init__(self, model_name: str, version: str):
        super().__init__(model_name, version, import_depth_estimation_model)

        self.bridge = CvBridge()

    def image_callback(self, img_msg):
        """ Take an image message and process it through the model

        Args:
            img_msg: Image message
        """
        img = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough')
        input_batch = self.transforms(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        out_img_msg = self.bridge.cv2_to_imgmsg(output, "32FC1")
        self.image_pub.publish(out_img_msg)

        print(output)


def main():
    """Main function to initialize the ROS node"""
    rospy.init_node("depth_estimation_node")

    model_name = rospy.get_param('~model_name', 'midas')
    version = rospy.get_param('~version', 'DPT_Large')

    depth_estimation_node = DepthEstimationNode(model_name, version)

    rospy.loginfo('depth_estimation is initialized')

    rospy.spin()


if __name__ == '__main__':
    main()
