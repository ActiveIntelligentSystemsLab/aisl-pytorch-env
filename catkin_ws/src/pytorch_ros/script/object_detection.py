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

from rospy_util.utils import imgmsg_to_pil, pil_to_imgmsg, write_on_pil

class ObjectDetectionNode(object):
    def __init__(self):

        # Get ROS params
        model_name = rospy.get_param('~model_name', 'yolov5s')
        version = rospy.get_param('~version', '18')

        # Preprocess image
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

        # Import model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rospy.loginfo(self.device)

        # Import YOLOv5 model from PyTorch Hub
        # Usage: https://github.com/ultralytics/yolov5/issues/36
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # Load ImageNet class names
        self.image_sub = rospy.Subscriber('image', Image, self.image_callback)
        self.image_pub = rospy.Publisher('visualize', Image, queue_size=10)
        self.result_pub = rospy.Publisher('detection_result', ObjectDetection, queue_size=10)
        self.bridge = CvBridge()

    def image_callback(self, img_msg):
        """ Take an image message and process it through the model

        Args:
            img_msg: Image message
        """
        pil_image, _, _ = imgmsg_to_pil(img_msg)

        with torch.no_grad():
            output = self.model(pil_image)
#            output = self.model(tensor_image.to(self.device))

        output.render()
        out_img = PIL.Image.fromarray(output.imgs[0])
        out_img_msg = pil_to_imgmsg(out_img)

        res_msg = self.result_to_msg(output.pandas().xyxy[0].to_json(orient="records"))

        self.result_pub.publish(res_msg)
        self.image_pub.publish(out_img_msg)

    def result_to_msg(self, results):
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

    obj_detection_node = ObjectDetectionNode()

    rospy.loginfo('object_detection_node is initialized')
    
    rospy.spin()  

if __name__=='__main__':
    main()