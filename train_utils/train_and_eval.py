import math
import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils
from train_utils.loss import dice_bce_loss
import torch.distributed as dist
import numpy as np
from medpy.metric.binary import hd95

def criterion(inputs, target):
    loss_= dice_bce_loss()
    loss =loss_( target, inputs)
    return loss

def soft_dice_coeff( y_true, y_pred,batch=True):
    smooth = 1.0
    if batch:
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
    else:
        i = y_true.sum(1).sum(1).sum(1)
        j = y_pred.sum(1).sum(1).sum(1)
        intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
    score = (2. * intersection + smooth) / (i + j + smooth)
    return score.mean()
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def evaluate(model, data_loader, device):
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    hd95_list = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iter_num = len(data_loader)
    print(iter_num)
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            torch.cuda.empty_cache()
            images, targets = images.to(device), targets.to(device)
            pet = torch.unsqueeze(images[:, 0, :, :], 1).repeat(1, 3, 1, 1)
            ct = torch.unsqueeze(images[:, 1, :, :], 1).repeat(1, 3, 1, 1)
            output = model(ct, pet)
            output = torch.sigmoid(output)
            pred_full = output.clone().detach().cpu().numpy()
            target = targets.clone().cpu().numpy()
            pred_full[pred_full >= 0.5] = 1
            pred_full[pred_full < 0.5] = 0

            gt_bin = target.astype(np.uint8)
            pred_bin = pred_full.astype(np.uint8)

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
                pred_bin[256, 256]=255
                hd = hd95(pred_bin, gt_bin)
            else:
                try:
                    hd = hd95(pred_bin, gt_bin)
                except:
                    hd = np.sqrt(gt_bin.shape[0]**2 + gt_bin.shape[1]**2)
            hd95_list.append(hd)

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

    return iou, dice, acc, mean_hd95

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pet = torch.unsqueeze(image[:, 0, :, :], 1).repeat(1,3,1,1)
            ct = torch.unsqueeze(image[:, 1, :, :], 1).repeat(1, 3, 1, 1)
            output = model(ct,pet)

            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
