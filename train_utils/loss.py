import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.nn import functional as F


class dice_bce_loss(nn.Module):
    def __init__(self, batch=False):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()


    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
    def dice_metric(self, y_true, y_pred):
        dice = self.soft_dice_coeff(y_true, y_pred)
        return dice
    def resize(self, y_true, h, w):
        b = y_true.shape[0]
        y = np.zeros((b, h,w ,y_true.shape[1]))

        y_true = np.array(y_true.cpu())
        for id in range(b):
            y1 = y_true[id,:,:,:].transpose(1,2,0)
            a = cv2.resize(y1, (h, w))
            if a.ndim == 2:
                a=np.expand_dims(a,axis=-1)
            y[id, :, :, :]=a
        y=y.transpose(0,3,1,2)
        return torch.Tensor(y)

    def __call__(self, y_true, y_pred):
        # the ground_truth map is resized to the resolution of the predicted map during training
        if y_true.shape[2] != y_pred.shape[2] or y_true.shape[3] != y_pred.shape[3]:
            y_true = self.resize(y_true, y_pred.shape[2], y_pred.shape[3]).cuda()

        loss1=F.binary_cross_entropy_with_logits(y_pred, y_true, reduce='none')
        y_pred = torch.sigmoid(y_pred)
        loss2 = self.soft_dice_loss(y_true, y_pred)
        return loss1+loss2

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, preds, labels):
        eps = 1e-7
        loss_y1 = -1 * self.alpha * \
                  torch.pow((1 - preds) , self.gamma) * \
                  torch.log(preds + eps) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds,
                                                    self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)

