import os
import time
import datetime
import random
import numpy as np
import torch
import cv2
import torch.nn as nn
from models.builder import EncoderDecoder as segmodel
from utils.logger import get_logger
import argparse
from easydict import EasyDict as edict
from medpy.metric.binary import hd95
from tqdm import tqdm

parser = argparse.ArgumentParser()
logger = get_logger()
C = edict()
config = C
C.backbone = 'sigma_tiny' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512
C.image_height=512
C.image_width =512
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.num_classes = 1

def processImage(pet_path, ct_path, mask_path, model, outPath, image_id, total_tp, total_fp, total_fn, total_tn, hd95_list):
    model.eval()
    pet_img = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)  #pet
    ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)  #ct
    # print(image_id)

    ct_img = np.expand_dims(ct_img, axis=2)
    pet_img = np.expand_dims(pet_img, axis=2)
    ct_img = np.array(ct_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    pet_img = np.array(pet_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    pet_img = pet_img[np.newaxis, :, :, :]
    ct_img = ct_img[np.newaxis, :, :, :]
    pet_img = torch.tensor(pet_img).cuda().repeat(1, 3, 1, 1)
    ct_img = torch.tensor(ct_img).cuda().repeat(1, 3, 1, 1)
    model.cuda()
    with torch.no_grad():
        pred =  torch.sigmoid(model.forward(ct_img,pet_img))
    pred = pred.cpu().numpy()
    pred = np.squeeze(pred, axis=0).transpose(1, 2, 0)*255.0
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(outPath + image_id+'.png', pred)

    gt_bin = (mask_img > 255*0.5).astype(np.uint8)
    pred_bin = (pred[:,:,0] > 255*0.5).astype(np.uint8)

    tp = np.sum((gt_bin == 1) & (pred_bin == 1))
    fp = np.sum((gt_bin == 0) & (pred_bin == 1))
    fn = np.sum((gt_bin == 1) & (pred_bin == 0))
    tn = np.sum((gt_bin == 0) & (pred_bin == 0))

    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        hd = 0.0
    elif gt_bin.sum() == 0 or pred_bin.sum() == 0:
        pred_bin[256, 256]=1
        hd = hd95(pred_bin, gt_bin)
    else:
        try:
            hd = hd95(pred_bin, gt_bin)
        except:
            hd = np.sqrt(gt_bin.shape[0]**2 + gt_bin.shape[1]**2)
    hd95_list.append(hd)

    return total_tp, total_fp, total_fn, total_tn, hd95_list


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    checkpoint = torch.load('save_model/CIPA.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    with open(os.path.join(args.split_train_val_test, 'test.txt'), 'r') as f:
        test_list = [x[:-1] for x in f]
    outPath="./results/"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    hd95_list = []
    for image_id in tqdm(test_list):
        pet_path = os.path.join(args.img_dir, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, "_PET.png")
        ct_path = os.path.join(args.img_dir, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, "_CT.png")
        mask_path = os.path.join(args.img_dir, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, "_mask.png")
        total_tp, total_fp, total_fn, total_tn, hd95_list = processImage(pet_path,ct_path,mask_path,model,outPath,image_id,total_tp, total_fp, total_fn, total_tn,hd95_list)

    total_pixels = total_tp + total_fp + total_fn + total_tn
    # IoU
    denominator_iou = total_tp + total_fp + total_fn
    iou = 1.0 if denominator_iou == 0 else total_tp / denominator_iou
    # Dice/F1
    denominator_dice = 2 * total_tp + total_fp + total_fn
    dice = 1.0 if denominator_dice == 0 else (2 * total_tp) / denominator_dice
    # Accuracy
    acc = np.mean([total_tp/(total_tp+total_fn), total_tn/(total_tn+total_fp)])
    # HD95
    mean_hd95 = np.mean(hd95_list)
    print(f"IoU: {iou:.6f}")
    print(f"Dice: {dice:.6f}")
    print(f"Acc: {acc:.6f}")
    print(f"HD95: {mean_hd95:.3f}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--img_dir', type=str, default="data/PCLT20k/")
    parser.add_argument('--split_train_val_test', type=str, default='data/PCLT20k/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
