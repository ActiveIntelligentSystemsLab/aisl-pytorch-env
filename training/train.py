# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import json
from PIL import Image
import torch
from torchvision import transforms
from util.util import import_model
from dataset.ycb import SyntheticClassificationDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(model, data_loader, epoch, criterion, optimizer, scheduler):
    for ep in range(epoch):
        print("=== Start epoch {} ===".format(ep+1))
        loss_sum = 0.0
        for i, batch in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            image = batch[0].to('cuda')
            label = batch[1].to('cuda')

            output = model(image)

            loss = criterion(output, label)

            loss_sum += loss.item()

            loss.backward()
            optimizer.step()

        print("Loss avg: {}", loss_sum/len(data_loader))
        scheduler.step()

def main():
    trainset = SyntheticClassificationDataset(root="./data_list/")
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
    model = import_model(model_name="efficientnet", version="b7", num_classes=77)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    train(model, data_loader, 100, criterion, optimizer, scheduler)

if __name__=='__main__':
    main()