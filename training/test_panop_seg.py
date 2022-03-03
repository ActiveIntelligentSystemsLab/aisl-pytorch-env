# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import random
import cv2
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
import torch
from torchvision import transforms
from util.util import import_model
import detectron2
from detectron2.utils.logger import setup_logger
import argparse

setup_logger()


def get_arguments():
    """Get commandline arguments

    """
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-dir', type=str, default='/tmp/dataset/Cityscapes/',
                        help='Data location')
    parser.add_argument('--model', type=str, default='deeplab',
                        help='The name of the model')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='The version of the backbone')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use in computation')

    args = parser.parse_args()

    return args

# 画像表示用の関数を定義


def cv2_imshow(img):
    plt.figure(figsize=(8, 8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    plt.imshow(img)
#    plt.show()
    plt.imsave("pred.png", img)


def main():
    args = get_arguments()

    im = cv2.imread(os.path.join(args.data_dir, "leftImg8bit/test/berlin/berlin_000023_000019_leftImg8bit.png"))
    # Detectron2のコンフィグを読み込みます
    cfg = get_cfg()

    # モデル固有のコンフィグをマージします
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))

    # thresholdを設定します。この閾値より予測の確度が高いもののみ出力されます。
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # 今回利用するモデルのトレーニング済みファイルを読み込みます。
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

    # defaultだとcfgのDEVICE=cudaになっているので、cudaない場合はcpuに変更
    cfg.MODEL.DEVICE = args.device

    # predictorを構築し、予測を実行します
    predictor = DefaultPredictor(cfg)
    # Panoptic Segmentationは少しコード違います
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

    # 予測された結果を確認します。
    # 出力フォーマットはhttps://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-formatで確認できます
#    print(outputs["instances"].pred_classes)
#    print(outputs["instances"].pred_boxes)

    # `Visualizer`を使用することで、画像上に予測結果を表示できます
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    cv2_imshow(out.get_image()[:, :, ::-1])


if __name__ == '__main__':
    main()
