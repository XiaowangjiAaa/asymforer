'''
Our code is partially adapted from RedNet (https://github.com/JinDongJiang/RedNet)
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Fixed comma separator
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch import nn
from src.AsymFormer import B0_T
import NYUv2_dataloader as Data
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='RGB-Dcrackdataset')
parser.add_argument('--data-dir', default='./RGB-Dcrackdataset', metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt-dir', default='./model_M1/asym_o_cro_5e-5_1e-4', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480

# Create log directory if not exists
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)
log_txt_path = os.path.join(args.ckpt_dir, 'loss_log.txt')
curve_img_path = os.path.join(args.ckpt_dir, 'loss_curve.png')


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean', ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))
        targets = targets.view(-1).long()
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=4,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def save_loss_curve(epoch_losses, img_path):
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig(img_path)
    plt.close()
    print(f'Saved loss curve to {img_path}')


def train():
    setup_seed(2333)
    train_data = Data.RGBD_Dataset(transform=transforms.Compose([Data.scaleNorm(),
                                                                 Data.RandomScale((1.0, 1.4, 2.0)),
                                                                 Data.RandomHSV((0.9, 1.1),
                                                                                (0.9, 1.1),
                                                                                (25, 25)),
                                                                 Data.RandomCrop(image_h, image_w),
                                                                 Data.RandomFlip(),
                                                                 Data.ToTensor(),
                                                                 Data.Normalize()]),
                                   phase_train=True,
                                   data_dir=args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=False)

    num_train = len(train_data)
    model = B0_T(num_classes=2)
    # CEL_weighted = nn.BCEWithLogitsLoss()
    # CEL_weighted = FocalLoss(ignore_index=0)
    CEL_weighted = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    #CEL_weighted = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    model.to(device)
    CEL_weighted.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.weight_decay)
    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # Initialize loss tracking
    epoch_losses = []
    start_epoch = int(args.start_epoch)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        batch_count = 0

        for batch_idx, sample in enumerate(train_loader):
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label']]

            optimizer.zero_grad()
            out = model(image, depth)
            loss = CEL_weighted(out, target_scales[0].long())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            batch_count += 1
            global_step += 1

            if global_step % args.print_freq == 0 or global_step == 1:
                time_inter = time.time() - epoch_start_time
                processed_items = (batch_idx + 1) * args.batch_size
                print_log(global_step, epoch, processed_items, args.batch_size,
                          num_train, loss, time_inter)

        # Calculate epoch loss
        epoch_loss = running_loss / batch_count
        epoch_losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time

        # Save epoch loss to text file
        with open(log_txt_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}/{args.epochs} - Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f}s\n')

        print(f'Epoch {epoch + 1}/{args.epochs} completed. Avg Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f}s')

        # Save checkpoint periodically
        if (epoch + 1) % args.save_epoch_freq == 0 or epoch == args.epochs - 1:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch + 1,
                      batch_count, num_train)

    # Save final model and plot loss curve
    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs, 0, num_train)
    save_loss_curve(epoch_losses, curve_img_path)
    print(f"Training completed. Loss log saved to {log_txt_path}")


if __name__ == '__main__':
    train()