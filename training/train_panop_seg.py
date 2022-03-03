# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import argparse
import torch
from dataset.segmentation.cityscapes import AlbCityscapes, class_wts_list
from util.util import import_segmentation_model, visualize_segmentation, get_log_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from torchmetrics import JaccardIndex as IoU


def get_arguments():
    """Get commandline arguments

    """
    parser = argparse.ArgumentParser(description='Training parameters etc.')
    parser.add_argument('--data-dir', type=str, default='/tmp/dataset/Cityscapes/',
                        help='Data location')
    parser.add_argument('--log-root', type=str, default='/tmp/runs/cityscapes',
                        help='Log location')
    parser.add_argument('--model', type=str, default='deeplab',
                        help='The name of the model')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='The version of the backbone')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--epoch', type=int, default=200,
                        help='The number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use in computation')

    args = parser.parse_args()

    return args


def train(model, data_loader, epoch, criterion, optimizer, metric=None, writer=None, device='cuda'):
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
    model.train()
    for i, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        print("Train")
        optimizer.zero_grad()
        image = batch["rgb_img"].to(device)
        orig_img = batch["orig_img"]
        label = batch["label"].long().to(device)

        output = model(image)['out']

        loss = criterion(output, label)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        output_argmax = torch.argmax(output, dim=1)

        # Calculate metric
        if metric is not None:
            if i == 0:
                iou = metric(output_argmax, label)
                print(iou)
            else:
                iou += metric(output_argmax, label)

        if writer is not None:
            writer.add_scalar('train/loss', loss, epoch * len(data_loader) + i)

        output_argmax = torch.argmax(output, dim=1)
        if i == 0 and writer is not None:
            print("Visualize!")
            visualize_segmentation(orig_img, output_argmax, label, writer, epoch, is_train=True)

    return loss_sum / len(data_loader)


def val(model, data_loader, epoch, criterion, metric=None, writer=None, device='cuda'):
    """ Training for one epoch

    Args:
        model       : Network to train
        data_loader : Data loader
        epoch       : Current epoch number
        criterion   : Loss function

    Return:
        Averaged loss value
    """
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
            image = batch["rgb_img"].to(device)
            orig_img = batch["orig_img"]
            label = batch["label"].long().to(device)

            output = model(image)['out']

            loss = criterion(output, label)

            output_argmax = torch.argmax(output, dim=1)

            loss_sum += loss.item()

            # Calculate metric
            if metric is not None:
                if i == 0:
                    iou = metric(output_argmax, label)
                else:
                    iou += metric(output_argmax, label)

            if i == 0 and writer is not None:
                visualize_segmentation(orig_img, output_argmax, label, writer, epoch, is_train=False)

    return loss_sum / len(data_loader)


def main():
    args = get_arguments()

    #
    # Data transformations: Transformations to apply on the training data
    #
    train_transform = A.Compose(
        [
            A.Resize(height=512, width=1024),
            A.RandomScale(scale_limit=(0.3, 1.8)),
            A.RandomCrop(height=256, width=512),
            A.HorizontalFlip(p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=256, width=512),
        ]
    )

    #
    # Dataset: A class that loads images with augmentation
    #
    trainset = AlbCityscapes(root=args.data_dir, split='train', mode='fine',
                             target_type='semantic', alb_transforms=train_transform)
    valset = AlbCityscapes(root=args.data_dir, split='val', mode='fine',
                           target_type='semantic', alb_transforms=val_transform)
    num_classes = 19
    ignore_index = 255

    #
    # Data loader: A class to wrap a dataset and manages training batches
    #
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, pin_memory=False)

    #
    # Model: Network to train
    #
    model_name = "deeplab"
    model_ver = "resnet50"
    model = import_segmentation_model(model_name=args.model, version=args.backbone, num_classes=num_classes)
    model.to(args.device)

    #
    # Loss function: Target function to be minimized
    #
    class_wts = torch.Tensor(class_wts_list).to(args.device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_wts, ignore_index=ignore_index)
#    metric = IoU(num_classes=num_classes)
    metric = None

    #
    # Optimizer: Updates
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #
    # Scheduler: Gradually changes the learning rate
    #
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    log_path = get_log_path(args.log_root, model_name + "-" + model_ver)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    writer = SummaryWriter(log_path)
#    writer = None

    # =========
    # Training
    # =========
    for ep in range(args.epoch):
        print("=== Start epoch {} ===".format(ep+1))

        loss = train(model, train_loader, ep, criterion, optimizer, metric=metric, writer=writer, device=args.device)
        scheduler.step()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)

        print("Loss avg: {}".format(loss))

        # Validation
        if ep % 5 == 0:
            loss_val = val(model, val_loader, ep, criterion=criterion, metric=metric, writer=writer, device=args.device)
            writer.add_scalar('val/loss', loss_val, ep)

#        torch.save(model.state_dict(), os.path.join(log_path, model_name+'_'+model_ver+'_'+str(epoch)+'.pt'))


if __name__ == '__main__':
    main()
