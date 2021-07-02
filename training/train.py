# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import json
import os
from PIL import Image
import torch
from torchvision import transforms
from util.util import import_model, visualize_classification, get_log_path
from dataset.ycb import SyntheticClassificationDataset, ycb_class_list
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(model, data_loader, epoch, criterion, optimizer, writer):
    """ Training for one epoch

    Args:
        model       : Network to train
        data_loader : Data loader
        epoch       : Current epoch number
        criterion   : Loss function
        optimizer   : Optimizer

    Return:
        Averaged loss value
    """
    loss_sum = 0.0
    for i, batch in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        image = batch["rgb_img"].to('cuda')
        orig_img = batch["orig_img"]
        label = batch["label_id"].to('cuda')

        output = model(image)

        loss = criterion(output, label)

        loss_sum += loss.item()

        loss.backward()
        optimizer.step()

        # Visualize the first batch
        if i == 0:
            visualize_classification(image_batch=orig_img, output=output, class_list=ycb_class_list, writer=writer, epoch=epoch)

    return loss_sum /len(data_loader)

def main():
    #
    # Dataset: A class that loads images with augmentation
    #
    trainset = SyntheticClassificationDataset(root="./data_list/")

    #
    # Data loader: A class to wrap a dataset and manages training batches
    #
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)

    #
    # Model: Network to train
    #
    model_name = "efficientnet"
    model_ver  = "b0"
    model = import_model(model_name=model_name, version=model_ver, num_classes=77)
    model.to('cuda')

    #
    # Loss function: Target function to be minimized
    #
    criterion = torch.nn.CrossEntropyLoss()

    #
    # Optimizer: Updates 
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)
    
    #
    # Scheduler: Gradually changes the learning rate
    #
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    log_path = get_log_path("/tmp/runs/ycb_classification", model_name + "-" + model_ver)
    writer = SummaryWriter(log_path)

    #=========
    # Training
    #========= 
    epoch = 100
    for ep in range(epoch):
        print("=== Start epoch {} ===".format(ep+1))

        loss = train(model, data_loader, ep, criterion, optimizer, writer)
        print("Loss avg: {}", loss)
        writer.add_scalar('loss', loss, ep)
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(log_path, 'ycb_'+model_name+'_'+model_ver+'.pt'))

if __name__=='__main__':
    main()