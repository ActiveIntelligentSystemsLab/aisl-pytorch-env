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

# Other
from PIL import Image

from util.util import import_segmentation_model

from rospy_util.utils import imgmsg_to_pil, pil_to_imgmsg, write_on_pil
from rospy_util.inference_node_base import InferenceNodeBase


class SemanticSegmentationNode(InferenceNodeBase):
    def __init__(self, model_name: str, version: str):
        super().__init__(model_name, version, import_segmentation_model)

        self.bridge = CvBridge()

        # create a color pallette, selecting a color for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.colors = torch.as_tensor([i for i in range(21)])[
            :, None] * self.palette
        self.colors = (self.colors % 255).numpy().astype("uint8")

    def image_callback(self, img_msg):
        """ Take an image message and process it through the model

        Args:
            img_msg: Image message
        """
        pil_image, _, _ = imgmsg_to_pil(img_msg)
        tensor_image = self.transforms(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor_image)

        output_predictions = torch.argmax(output['out'][0], dim=0)

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(
            output_predictions.byte().cpu().numpy()).resize(pil_image.size)
        r.putpalette(self.colors)
        r = r.convert('RGB')

        # Alpha blend for
        r = Image.blend(pil_image, r, 0.5)

        out_img_msg = pil_to_imgmsg(r)

        self.image_pub.publish(out_img_msg)


def main():
    """Main function to initialize the ROS node"""
    rospy.init_node("semantic_segmentation_node")

    model_name = rospy.get_param('~model_name', 'deeplabv3')
    version = rospy.get_param('~version', 'mobilenet_v3_large')

    semantic_segmentation_node = SemanticSegmentationNode(model_name, version)

    rospy.loginfo('semantic_segmentation is initialized')

    rospy.spin()


if __name__ == '__main__':
    main()
