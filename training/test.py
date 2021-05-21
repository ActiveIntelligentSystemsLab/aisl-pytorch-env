# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import json
from PIL import Image
import torch
from torchvision import transforms
from util.util import import_model

def main():

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open('img.jpg')).unsqueeze(0)
    print(img.shape) # torch.Size([1, 3, 224, 224])

    # Load ImageNet class names
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]


    # Show all available models in torch.hub
    print("Available models:")
    print(torch.hub.list('pytorch/vision:v0.6.0'))
    print()
    print("Model name?: ")
    model_name = input()
    print("Version?: ")
    version = input()
    # Import model
    model = import_model(model_name, version)
    model.to('cuda')

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img.to('cuda'))

    # Print predictions
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

if __name__=='__main__':
    main()