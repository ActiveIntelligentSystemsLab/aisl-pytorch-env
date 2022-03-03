# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import torch
import cv2
import numpy as np
import torchvision
import os
from collections import OrderedDict


def import_model(model_name, version, num_classes=-1):
    """ Import pre-trained models
    https://tutorials.pytorch.kr/beginner/finetuning_torchvision_models_tutorial.html

    Args:
        model_name: Name of the model to import
            ['resnet','alexnet','inception','efficientnet','mobilenet','squeezenet','densenet']
        version: Version of the model (It depends on the model)

    Return:
        Imported model
    """
    model = None
    # ResNet
    if model_name == 'resnet':
        if version in ['18', '34', '50', '101', '152']:
            model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet'+version, pretrained=True)
            if num_classes > 0:
                if version in ['18', '34']:
                    model.fc = torch.nn.Linear(512, num_classes)
                elif version in ['50', '101', '152']:
                    model.fc = torch.nn.Linear(2048, num_classes)
        else:
            print("There's no such ('{}') version of ResNet.".format(version))
            raise ValueError

    # AlexNet
    elif model_name == 'alexnet':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)

        if num_classes > 0:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)

    # Inception
    elif model_name == 'inception':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)

        if num_classes > 0:
            model.AuxLogits.fc = torch.nn.Linear(768, num_classes)
            model.fc = torch.nn.Linear(2048, num_classes)

    # EfficientNet
    elif model_name == 'efficientnet':
        from efficientnet_pytorch import EfficientNet

        if version in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
            model = EfficientNet.from_pretrained('efficientnet-'+version, num_classes=num_classes if num_classes > 0 else 1000)
        else:
            print("There's no such ('{}') version of EfficientNet.".format(version))
            raise ValueError

    # MobileNet v2
    elif model_name == 'mobilenet':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

        if num_classes > 0:
            self.classifier[1] = model.nn.Linear(model.last_channel, num_classes)

    # SqueezeNet
    elif model_name == 'squeezenet':
        if version in ['1_0', '1_1']:
            model = torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet'+version, pretrained=True)
        else:
            print("There's no such ('{}') version of SqueezeNet".format(version))
            raise ValueError

        if num_classes > 0:
            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    elif model_name == 'densenet':
        if version in ['121', '161', '169', '201']:
            model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet'+version, pretrained=True)
        else:
            print("There's no such ('{}') version of DenseNet".format(version))
            raise ValueError

        if num_classes > 0:
            model.classifier = torch.nn.Linear(1024, num_classes)
    else:
        raise ValueError

    return model


def import_detection_model(model_name, version, num_classes=-1):
    """ Import pre-trained object detection models
    https://tutorials.pytorch.kr/beginner/finetuning_torchvision_models_tutorial.html

    Args:
        model_name: Name of the model to import
            ['ssd', 'yolov5']
        version: Version of the model (It depends on the model)

    Return:
        Imported model
    """
    model = None
    # ResNet
    if model_name == 'ssd':
        pass
    elif True:
        pass


def import_segmentation_model(model_name, version, num_classes=-1):
    """ Import segmentation models
    https://tutorials.pytorch.kr/beginner/finetuning_torchvision_models_tutorial.html

    Args:
        model_name: Name of the model to import
            ['deeplab']
        version: Version of the model (It depends on the model)

    Return:
        Imported model
    """
    if model_name == 'deeplab':
        if version in ['resnet50', 'resnet101', 'mobilenet_v3_large']:
            #            model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet'+version, pretrained=True)
            model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_'+version, pretrained=True)

        if num_classes > 0:
            model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError

    return model


def visualize_classification(image_batch, output, class_list, writer, epoch):
    """ Visualize the prediction results

    Args:
        image_batch : Input image batch
        output      : Output of the model for the image batch
        class_list  : List that stores class names. ith element stores the name of label i
        writer      : Writer of Tensorboard
        epoch       : The number of the current epoch
    """
    for i in range(image_batch.size()[0]):
        # Convert i the element of the batch to a NumPy array (= OpenCV image)
        cv_image = image_batch[i].cpu().numpy().transpose(1, 2, 0)
        cv_image = (cv_image * 255).astype(np.uint8)
        cv_image = np.ascontiguousarray(cv_image)

        # Get the name of the predicted class
        label = torch.argmax(output[i])
        label_name = class_list[label.item()]

        # Write the label name in the image
        # Show boadered characters for visibility
        # https://imagingsolution.net/program/draw-outline-character/ (Japanese)
        #   Write contour
        cv2.putText(cv_image, label_name, (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 5, cv2.LINE_AA)
        #   Write inside
        cv2.putText(cv_image, label_name, (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 0), 2, cv2.LINE_AA)

        # Append the image in the new batch
        if i == 0:
            np_batch = np.expand_dims(cv_image.transpose(2, 0, 1), axis=0)
        else:
            np_batch = np.append(np_batch, np.expand_dims(cv_image.transpose(2, 0, 1), axis=0), axis=0)

    # Make a grid image and write it in the log
    grid = torchvision.utils.make_grid(torch.from_numpy(np_batch)).numpy()
    writer.add_image('results', grid, epoch)


def visualize_segmentation(image_batch, output, gt_label, writer, epoch, is_train=True):
    """ Visualize the prediction results

    Args:
        image_batch : Input image batch
        output      : Output of the model for the image batch
        gt_label    : Ground truth label map
        writer      : Writer of Tensorboard
        epoch       : The number of the current epoch
    """

    # Do transformation of label tensor and prediction tensor
    from dataset.segmentation.cityscapes import color_encoding
    gt_label = batch_transform(gt_label.data.cpu(), LongTensorToRGBPIL(color_encoding))
    output = batch_transform(output.cpu(), LongTensorToRGBPIL(color_encoding))

    # Make a grid image and write it in the log
    input_grid = torchvision.utils.make_grid(image_batch.data.cpu()).numpy()
    gt_grid = torchvision.utils.make_grid(gt_label).numpy()
    pred_grid = torchvision.utils.make_grid(output).numpy()

    data = 'train' if is_train else 'val'

    writer.add_image('{}/input'.format(data), input_grid, epoch)
    writer.add_image('{}/gt'.format(data), gt_grid, epoch)
    writer.add_image('{}/pred'.format(data), pred_grid, epoch)


def get_log_path(root, net_type):
    """Create a directory to the log files (Tensorboard etc.)

    Args:
        root (string): A path of the root directory
        net_type (string): The type of the generator network used in the training

    Returns:
        string: A path to the directory
    """
    import datetime
    now = datetime.datetime.now()
    now += datetime.timedelta(hours=9)
    timestr = now.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(root, net_type+"_"+timestr)

    return log_path


def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.
    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``
    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


class LongTensorToRGBPIL(object):
    """Converts a ``torch.LongTensor`` to a ``PIL image``.

    The input is a ``torch.LongTensor`` where each pixel's value identifies the
    class.

    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.

    """

    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``

        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert

        Returns:
        A ``PIL.Image``.

        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)

        # Initialize
        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

#        return ToPILImage()(color_tensor)
        return color_tensor
