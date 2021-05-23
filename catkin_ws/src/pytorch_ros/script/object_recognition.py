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
import cv2
import copy
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import struct

#
# imgmsg_to_pil and pil_to_imgmsg are from 
#   https://github.com/AndreaCensi/ros_node_utils/blob/master/src/ros_node_utils/conversions/np_images.py
#
def imgmsg_to_pil(
    rosimage,
    encoding_to_mode={
        # not sure http://answers.ros.org/question/46746/need-help-with-accessing-the-kinect-depth-image-using-opencv/
        "16UC1": "L",
        "bayer_grbg8": "L",
        "mono8": "L",
        "8UC1": "L",
        "8UC3": "RGB",
        "rgb8": "RGB",
        "bgr8": "RGB",
        "rgba8": "RGBA",
        "bgra8": "RGBA",
        "bayer_rggb": "L",
        "bayer_gbrg": "L",
        "bayer_grbg": "L",
        "bayer_bggr": "L",
        "yuv422": "YCbCr",
        "yuv411": "YCbCr",
    },
    PILmode_channels={"L": 1, "RGB": 3, "RGBA": 4, "YCbCr": 3},
):
    conversion = "B"
    channels = 1
    if rosimage.encoding.find("32FC") >= 0:
        conversion = "f"
        channels = int(rosimage.encoding[-1])
    elif rosimage.encoding.find("64FC") >= 0:
        conversion = "d"
        channels = int(rosimage.encoding[-1])
    elif rosimage.encoding.find("8SC") >= 0:
        conversion = "b"
        channels = int(rosimage.encoding[-1])
    elif rosimage.encoding.find("8UC") >= 0:
        conversion = "B"
        channels = int(rosimage.encoding[-1])
    elif rosimage.encoding.find("16UC") >= 0:
        conversion = "H"
        channels = int(rosimage.encoding[-1])
    elif rosimage.encoding.find("16SC") >= 0:
        conversion = "h"
        channels = int(rosimage.encoding[-1])
    elif rosimage.encoding.find("32UC") >= 0:
        conversion = "I"
        channels = int(rosimage.encoding[-1])
    elif rosimage.encoding.find("32SC") >= 0:
        conversion = "i"
        channels = int(rosimage.encoding[-1])
    else:
        if rosimage.encoding.find("rgb") >= 0 or rosimage.encoding.find("bgr") >= 0:
            channels = 3

    data = struct.unpack(
        (">" if rosimage.is_bigendian else "<")
        + "%d" % (rosimage.width * rosimage.height * channels)
        + conversion,
        rosimage.data,
    )

    if conversion == "f" or conversion == "d":
        dimsizes = [rosimage.height, rosimage.width, channels]
        imagearr = numpy.array(255 * I, dtype=numpy.uint8)  # @UndefinedVariable
        im = PIL.Image.frombuffer(
            "RGB" if channels == 3 else "L",
            dimsizes[1::-1],
            imagearr.tostring(),
            "raw",
            "RGB",
            0,
            1,
        )
        if channels == 3:
            im = PIL.Image.merge("RGB", im.split()[-1::-1])
        return im, data, dimsizes
    else:
        if not rosimage.encoding in encoding_to_mode:
            msg = "Could not find %s in %s" % (
                rosimage.encoding,
                list(encoding_to_mode.keys()),
            )
            raise ValueError(msg)
        if rosimage.encoding in ["16UC1", "bayer_grbg8"]:
            warnings.warn("Probably conversion not correct for %s" % rosimage.encoding)
        mode = encoding_to_mode[rosimage.encoding]

        step_size = PILmode_channels[mode]
        dimsizes = [rosimage.height, rosimage.width, step_size]
        im = PIL.Image.frombuffer(
            mode, dimsizes[1::-1], rosimage.data, "raw", mode, 0, 1
        )
#        if mode == "RGB":
#            im = PIL.Image.merge("RGB", im.split()[-1::-1])
        return im, data, dimsizes
    
def pil_to_imgmsg(
    image,
    encodingmap={"L": "mono8", "RGB": "rgb8", "RGBA": "rgba8", "YCbCr": "yuv422"},
    PILmode_channels={"L": 1, "RGB": 3, "RGBA": 4, "YCbCr": 3},
):
    # import roslib  # @UnresolvedImport @UnusedImport
    # import rospy  # @UnresolvedImport @UnusedImport

    # from sensor_msgs.msg import CompressedImage  # @UnusedImport @UnresolvedImport

    rosimage = Image()
    # adam print 'Channels image.mode: ',PILmode_channels[image.mode]
    rosimage.encoding = encodingmap[image.mode]
    (rosimage.width, rosimage.height) = image.size
    rosimage.step = PILmode_channels[image.mode] * rosimage.width
    rosimage.data = image.tobytes()
    return rosimage

def write_on_pil(pil_image, text):
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.text((0,0), text)

class ObjectRecognitionNode(object):
    def __init__(self):

        # Get ROS params
        model_name = rospy.get_param('~model_name', 'resnet')
        version = rospy.get_param('~version', '18')
        label_map_path = rospy.get_param('~label_map_path', '')

        # Preprocess image
        self.transforms = transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

        # Import model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rospy.loginfo(self.device)
        self.model = import_model(model_name, version)
        self.model.to(self.device)
        self.model.eval()

        # Load ImageNet class names
        self.labels_map = json.load(open(label_map_path))
        self.labels_map = [self.labels_map[str(i)] for i in range(1000)]

        self.image_sub = rospy.Subscriber('image', Image, self.image_callback)
        self.image_pub = rospy.Publisher('visualize', Image, queue_size=10)
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
            res_str = '{label:<75} ({p:.2f}%)'.format(label=self.labels_map[idx], p=prob*100)
            if is_top:
                write_on_pil(pil_image, res_str)
                img_msg_pub = pil_to_imgmsg(pil_image)
                rospy.loginfo(pil_image.mode)
                # Publish image for visualization and the label of the top object
                self.image_pub.publish(img_msg_pub)
                self.result_pub.publish(res_str)
                is_top = False

def main():
    """Main function to initialize the ROS node"""
    rospy.init_node("object_recognition_node")

    obj_recog_node = ObjectRecognitionNode()

    rospy.loginfo('object_recognition_node is initialized')
    
    rospy.spin()  

if __name__=='__main__':
    main()