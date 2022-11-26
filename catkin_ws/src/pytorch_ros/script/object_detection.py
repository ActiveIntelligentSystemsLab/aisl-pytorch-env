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
from sensor_msgs.msg import RegionOfInterest
from cv_bridge import CvBridge
from deep_learning_msgs.msg import ObjectDetection

# PyTorch related
import torch
from torchvision import transforms
from util.util import import_model

# Other
import json
import cv2
import copy
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import struct

from util.util import import_detection_model

from rospy_util.utils import imgmsg_to_pil, pil_to_imgmsg, write_on_pil
from rospy_util.inference_node_base import InferenceNodeBase


class ObjectDetectionNode(InferenceNodeBase):
    def __init__(self, model_name: str, version: str):
        super().__init__(model_name, version, import_detection_model)

        # Additional publisher
        self.result_pub = rospy.Publisher(
            'detection_result', ObjectDetection, queue_size=10)
        self.bridge = CvBridge()

    def image_callback(self, img_msg):
        """ Take an image message and process it through the model

        Args:
            img_msg: Image message
        """
        pil_image, _, _ = imgmsg_to_pil(img_msg)

        with torch.no_grad():
            output = self.model(pil_image)
            print(output)

        output.render()
        # out_img = PIL.Image.fromarray(output.imgs[0])
        out_img = PIL.Image.fromarray(output.ims[0])
        out_img_msg = pil_to_imgmsg(out_img)

        res_msg = self.result_to_msg(
            output.pandas().xyxy[0].to_json(orient="records"))
        res_msg.header.frame_id = img_msg.header.frame_id

        self.result_pub.publish(res_msg)
        self.image_pub.publish(out_img_msg)

    def result_to_msg(self, results):
        """Convert result from YOLOv5 to ROS message

        Parameters
        ----------
        results: 
            Results 

        Returns
        -------
        res: `ObjectDetection`
            Message to store object detection results
        """
        results = json.loads(results)
        res = ObjectDetection()
        res.header.stamp = rospy.Time.now()
        for i, result in enumerate(results):
            # Create ROI instance
            roi = RegionOfInterest()
            roi.x_offset = int(result['xmin'])
            roi.y_offset = int(result['ymin'])
            roi.width = int(result['xmax']-result['xmin'])
            roi.height = int(result['ymax']-result['ymin'])

            res.boxes.append(roi)
            res.class_ids.append(result['class'])
            res.class_names.append(result['name'])
            res.scores.append(result['confidence'])

        return res


def main():
    """Main function to initialize the ROS node"""
    rospy.init_node("object_recognition_node")

    model_name = rospy.get_param('~model_name', 'yolov5s')
    version = rospy.get_param('~version', '18')

    obj_detection_node = ObjectDetectionNode(model_name, version)

    rospy.loginfo('object_detection_node is initialized')

    rospy.spin()


if __name__ == '__main__':
    main()
