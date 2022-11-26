# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import torch
import cv2
import numpy as np
import torchvision
import os


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
            model = torch.hub.load(
                'pytorch/vision:v0.9.1', 'resnet'+version, pretrained=True)
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
        model = torch.hub.load('pytorch/vision:v0.9.1',
                               'alexnet', pretrained=True)

        if num_classes > 0:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)

    # Inception
    elif model_name == 'inception':
        model = torch.hub.load('pytorch/vision:v0.9.1',
                               'inception_v3', pretrained=True)

        if num_classes > 0:
            model.AuxLogits.fc = torch.nn.Linear(768, num_classes)
            model.fc = torch.nn.Linear(2048, num_classes)

    # EfficientNet
    elif model_name == 'efficientnet':
        from efficientnet_pytorch import EfficientNet

        if version in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
            model = EfficientNet.from_pretrained(
                'efficientnet-'+version, num_classes=num_classes if num_classes > 0 else 1000)
        else:
            print("There's no such ('{}') version of EfficientNet.".format(version))
            raise ValueError

    # MobileNet v2
    elif model_name == 'mobilenet':
        model = torch.hub.load('pytorch/vision:v0.9.1',
                               'mobilenet_v2', pretrained=True)

        if num_classes > 0:
            self.classifier[1] = model.nn.Linear(
                model.last_channel, num_classes)

    # SqueezeNet
    elif model_name == 'squeezenet':
        if version in ['1_0', '1_1']:
            model = torch.hub.load(
                'pytorch/vision:v0.9.1', 'squeezenet'+version, pretrained=True)
        else:
            print("There's no such ('{}') version of SqueezeNet".format(version))
            raise ValueError

        if num_classes > 0:
            model.classifier[1] = torch.nn.Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    elif model_name == 'densenet':
        if version in ['121', '161', '169', '201']:
            model = torch.hub.load(
                'pytorch/vision:v0.9.1', 'densenet'+version, pretrained=True)
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
            ['ssd', 'yolov5s']
        version: Version of the model (It depends on the model)

    Return:
        Imported model
    """
    model = None
    # ResNet
    if model_name == 'yolov5s':
        model = torch.hub.load(
            'ultralytics/yolov5', model_name, pretrained=True)
    # elif model_name == 'ssd':
    #     pass
    else:
        raise ValueError

    return model


def import_segmentation_model(model_name, version, num_classes=-1):
    """ Import pre-trained object detection models
    https://tutorials.pytorch.kr/beginner/finetuning_torchvision_models_tutorial.html

    Args:
        model_name: Name of the model to import
            ['deeplabv3']
        version: Version of the model (It depends on the model)
            Deeplabv3: ['resnet101', 'resnet50', 'mobilenet_v3_large']

    Return:
        Imported model
    """
    model = None
    # ResNet
    if model_name == 'deeplabv3':
        model = torch.hub.load(
            'pytorch/vision:v0.9.1',
            model_name + "_" + version,
            pretrained=True)
    # elif model_name == 'ssd':
    #     pass
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
            np_batch = np.append(np_batch, np.expand_dims(
                cv_image.transpose(2, 0, 1), axis=0), axis=0)

    # Make a grid image and write it in the log
    grid = torchvision.utils.make_grid(torch.from_numpy(np_batch)).numpy()
    writer.add_image('results', grid, epoch)


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
