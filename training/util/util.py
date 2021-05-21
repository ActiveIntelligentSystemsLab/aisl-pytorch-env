# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import torch

def import_model(model_name, version):
    """ Import pre-trained models

    Args:
        model_name: Name of the model to import
            ['resnet','alexnet','inception','efficientnet','mobilenet','squeezenet','densenet']
        version: Version of the model (It depends on the model)

    Return:
        Imported model
    """
    # ResNet
    if model_name == 'resnet':
        if version == '18':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        elif version == '34':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        elif version == '101':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
        elif version == '152':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
        else:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    # AlexNet
    elif model_name == 'alexnet':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

    # Inception
    elif model_name == 'inception':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)

    # EfficientNet
    elif model_name == 'efficientnet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0')

    # MobileNet v2
    elif model_name == 'mobilenet':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)

    # SqueezeNet
    elif model_name == 'squeezenet':
        if version == '1_0':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
        else:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_1', pretrained=True)

    elif model_name == 'densenet':
        if version == '121':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        elif version == '161':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet161', pretrained=True)
        elif version == '169':
            model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169', pretrained=True)
        else:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True)
    else:
        raise ValueError
        
    return model