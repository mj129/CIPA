import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from train_utils import (train_one_epoch, evaluate, init_distributed_mode, save_on_master, mkdir,
                         create_lr_scheduler)
from torch.utils.data.distributed import DistributedSampler
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from utils.PCLT_dataset import  prepare_PETCT_dataset
import torch.distributed as dist
from models.builder import EncoderDecoder as segmodel
from utils.init_func import init_weight, group_weight
from utils.logger import get_logger
import argparse
import shutil
from easydict import EasyDict as edict
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

def params_count(model):
    return np.sum([p.numel() for p in model.parameters() if p.requires_grad]).item()

def create_model():
    model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    parameter = params_count(model)
    print('parameter:', parameter)
    return model

def main(args):
    if args.distributed:
        init_distributed_mode(args)
        print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset,val_dataset  = prepare_PETCT_dataset(args,transforms=True)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,shuffle=False)

    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,pin_memory=True,
        collate_fn=train_dataset.collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=1,
                                          num_workers=num_workers,
                                          pin_memory=True,
                                          shuffle=False,
                                          drop_last=False,
                                         collate_fn=val_dataset.collate_fn,
                                          )
    model = create_model()
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, args.lr)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer =='Adam':
        optimizer = torch.optim.Adam(params=params_to_optimize, lr=args.lr)
    elif args.optimizer =='AdamW':
        optimizer = torch.optim.AdamW(params_list, args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr= args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=args.warm_up,warmup_epochs=args.warm_up_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda'))
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    best_iou=0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        iou, dice, acc, mean_hd95 = evaluate(model, val_loader, device=device)

        print(f"IoU: {iou:.6f}")
        print(f"Dice: {dice:.6f}")
        print(f"Acc: {acc:.6f}")
        print(f"HD95: {mean_hd95:.3f}")

        save_file = {"model": model_without_ddp.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     # "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        if args.save_best is True:
            if best_iou < iou:
                best_iou = iou
                torch.save(save_file, os.path.join(WEIGHT_SAVE_DIR,"best_model.pth"))
            else:
                continue
        if args.save_best is False:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    print(f"best_metric: giou:{best_iou}")
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--eps', default=1e-8, type=float, help='adam eps')

    parser.add_argument('--lr', default=0.00006, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--img_dir', type=str, default="data/PCLT20k/")
    parser.add_argument('--split_train_val_test', type=str, default='data/PCLT20k/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument('--sync_bn', type=bool, default=True, help='whether using SyncBatchNorm')
    parser.add_argument("--warm_up", default=False, type=bool)
    parser.add_argument("--warm_up_epoch", default=5, type=int)
    parser.add_argument("--wandb", default=False, type=bool)
    args = parser.parse_args()

    return args

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] =":4096: 8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.benchmark =False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir,
                                   time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) )
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.mkdir(WEIGHT_SAVE_DIR)
    main(args)
