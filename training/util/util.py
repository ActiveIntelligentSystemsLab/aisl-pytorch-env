# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import torch

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
            model.classifier[6] = torch.nn.Linear(4096,num_classes)

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
            model = EfficientNet.from_pretrained('efficientnet-'+version, num_classes=num_classes if num_classes>0 else 1000)
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
            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

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