import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.AsymFormer_GSSA_full import B0_T
import NYUv2_dataloader as Data
from utils.utils import intersectionAndUnion, accuracy, AverageMeter, macc


IMAGE_H = 480
IMAGE_W = 640


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled):
    try:
        return torch.amp.autocast('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AsymFormer-GSSA (full)")
    # data
    parser.add_argument("--data-dir", type=str, default="./RGB-Dcrackdataset")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--amp", action="store_true", help="enable mixed precision")
    # loss
    parser.add_argument("--loss", type=str, default="ce_dice", choices=["ce", "ce_dice", "focal"])
    parser.add_argument("--ignore-index", type=int, default=-1,
                        help="忽略的标签值；-1 表示不忽略（背景参与损失，避免全预测裂缝的退化解）")
    parser.add_argument("--crack-weight", type=float, default=0.7,
                        help="CE 中裂缝类权重(0~1)，默认 0.7；用于缓解类别不平衡")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    # logging / checkpoint
    parser.add_argument("--ckpt-dir", type=str, default="./model_M1/gssa_full")
    parser.add_argument("--save-epoch-freq", type=int, default=5)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="每 N 个 epoch 验证一次并保存预测图（0 关闭）")
    parser.add_argument("--save-pred-max", type=int, default=100,
                        help="每次验证最多保存多少张预测可视化图（0 表示不保存）")
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            valid = (targets != self.ignore_index).float()
            return loss.sum() / (valid.sum() + 1e-10)
        return loss.mean() if self.reduction == "mean" else loss.sum()


class DiceLoss(nn.Module):
    """二值 Dice 损失（针对裂缝类），对类别不平衡鲁棒。"""

    def __init__(self, crack_class=1, eps=1e-6):
        super().__init__()
        self.crack_class = crack_class
        self.eps = eps

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)[:, self.crack_class]
        tgt = (targets == self.crack_class).float()
        inter = (probs * tgt).sum()
        return 1 - (2 * inter + self.eps) / (probs.sum() + tgt.sum() + self.eps)


class CEDiceLoss(nn.Module):
    """CE（含背景，可选类别权重）+ Dice，避免 ignore_index=0 导致的"全预测裂缝"退化解。"""

    def __init__(self, class_weights=None, ignore_index=-1, ce_weight=1.0,
                 dice_weight=1.0, crack_class=1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(crack_class)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        loss = self.ce_weight * self.ce(inputs, targets)
        loss = loss + self.dice_weight * self.dice(inputs, targets)
        return loss


def build_criterion(args):
    crack_w = getattr(args, "crack_weight", None)
    class_weights = None
    if crack_w is not None:
        class_weights = torch.tensor([1.0 - crack_w, crack_w], dtype=torch.float32)

    if args.loss == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_index)
    if args.loss == "ce_dice":
        return CEDiceLoss(class_weights=class_weights, ignore_index=args.ignore_index)
    return FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha,
                     ignore_index=args.ignore_index)


def build_lr_scheduler(optimizer, num_steps, args):
    warmup_steps = args.warmup_epochs * num_steps
    total_steps = args.epochs * num_steps

    def lr_factor(step):
        if step < warmup_steps:
            alpha = step / max(warmup_steps, 1)
            return args.min_lr / args.lr * (1 - alpha) + alpha
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max((1 - progress) ** 0.9, args.min_lr / args.lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def build_train_loader(args):
    transform = T.Compose([
        Data.scaleNorm(),
        Data.RandomScale((1.0, 1.4, 2.0)),
        Data.RandomHSV((0.9, 1.1), (0.9, 1.1), (25, 25)),
        Data.RandomCrop(IMAGE_H, IMAGE_W),
        Data.RandomFlip(),
        Data.ToTensor(),
        Data.Normalize(),
    ])
    dataset = Data.RGBD_Dataset(transform=transform, phase_train=True, data_dir=args.data_dir)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.workers, pin_memory=True, drop_last=True)


class SaveOrigin(object):
    """在 Normalize 之前保存原始(0~255)图像，供验证可视化使用。"""

    def __call__(self, sample):
        sample["origin_image"] = sample["image"].clone()
        return sample


def build_val_loader(args):
    transform = T.Compose([Data.scaleNorm(), Data.ToTensor(), SaveOrigin(), Data.Normalize()])
    dataset = Data.RGBD_Dataset(transform=transform, phase_train=False,
                                data_dir=args.data_dir, txt_name="test.txt")
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      num_workers=0, pin_memory=True)


def save_ckpt(path, model, optimizer, epoch, global_step, best_miou):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
    }, path)
    print("[save] {}".format(path))


def load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), ckpt.get("best_miou", 0.0)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, args, global_step):
    model.train()
    total_loss = 0.0
    nb = len(loader)
    start = time.time()
    pbar = tqdm(loader, desc="Epoch {}/{}".format(epoch + 1, args.epochs),
                total=nb, leave=True, ncols=120)

    for i, sample in enumerate(pbar):
        image = sample["image"].to(device)
        depth = sample["depth"].to(device)
        label = sample["label"].to(device).long()

        optimizer.zero_grad()
        if args.amp:
            with autocast_context(True):
                out = model(image, depth)
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(image, depth)
            loss = criterion(out, label)
            loss.backward()
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        scheduler.step()
        global_step += 1
        total_loss += loss.item()
        pbar.set_postfix(loss="{:.4f}".format(loss.item()),
                         avg="{:.4f}".format(total_loss / (i + 1)),
                         lr="{:.2e}".format(optimizer.param_groups[0]["lr"]))

    elapsed = time.time() - start
    return total_loss / nb, global_step, elapsed


@torch.no_grad()
def validate(model, loader, num_classes, device):
    model.eval()
    acc_meter = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    for sample in loader:
        image = sample["image"].to(device)
        depth = sample["depth"].to(device)
        label = sample["label"].numpy()
        pred = model(image, depth)
        output = torch.argmax(pred, 1).squeeze(0).cpu().numpy()
        acc, pix = accuracy(output, label)
        inter, union = intersectionAndUnion(output, label, num_classes)
        acc_meter.update(acc, pix)
        inter_meter.update(inter)
        union_meter.update(union)
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    return iou.mean(), acc_meter.average()


@torch.no_grad()
def validate_and_save(model, loader, num_classes, device, save_dir=None, max_save=100):
    """推理 + 计算完整指标(mIoU/Dice/clDice/...) + 保存预测可视化图像。

    保存格式: [原图 | GT(裂缝白) | 预测(裂缝白)] 横向拼接。
    返回指标 dict（含 'miou'、'dice_crack'、'cldice_crack' 等）。
    """
    from metrics import SegMetrics
    model.eval()
    m = SegMetrics(num_classes, crack_class=1)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, sample in enumerate(loader):
        image = sample["image"].to(device)
        depth = sample["depth"].to(device)
        label = sample["label"].numpy()
        pred = model(image, depth)
        pred_np = torch.argmax(pred, 1).cpu().numpy().squeeze(0).astype("int64")
        label_np = label.squeeze(0).astype("int64")
        m.update(pred_np, label_np)

        if save_dir is not None and i < max_save and "origin_image" in sample:
            origin = sample["origin_image"].squeeze(0).permute(1, 2, 0).cpu().numpy()
            origin = np.clip(origin, 0, 255).astype(np.uint8)
            gt = np.repeat(((label_np > 0).astype(np.uint8) * 255)[:, :, None], 3, axis=2)
            pd = np.repeat((pred_np.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
            vis = np.concatenate([origin, gt, pd], axis=1)
            cv2.imwrite(os.path.join(save_dir, "{:05d}.png".format(i)), vis)

    return m.summary()


def plot_loss(losses, path):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, "b-", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.ckpt_dir, exist_ok=True)
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_train_loader(args)
    val_loader = build_val_loader(args) if args.eval_every > 0 else None

    model = B0_T(num_classes=args.num_classes).to(device)
    criterion = build_criterion(args).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, len(train_loader), args)
    scaler = make_scaler(args.amp)

    start_epoch = 0
    global_step = 0
    best_miou = 0.0
    if args.resume:
        start_epoch, global_step, best_miou = load_ckpt(args.resume, model, optimizer, device)
        start_epoch += 1

    log_path = os.path.join(args.ckpt_dir, "loss_log.txt")
    epoch_losses = []

    for epoch in range(start_epoch, args.epochs):
        avg_loss, global_step, elapsed = train_one_epoch(model, train_loader, criterion, optimizer,
                                                         scheduler, scaler, device, epoch, args, global_step)
        epoch_losses.append(avg_loss)
        eta = (args.epochs - (epoch + 1)) * elapsed
        print("Epoch {}/{} loss {:.6f} | 用时 {:.1f}s | 剩余约 {:.1f}s".format(
            epoch + 1, args.epochs, avg_loss, elapsed, eta))

        with open(log_path, "a") as f:
            f.write("Epoch {}/{} loss {:.6f} time {:.1f}s\n".format(epoch + 1, args.epochs, avg_loss, elapsed))

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            pred_dir = os.path.join(args.ckpt_dir, "predictions", "epoch_{:03d}".format(epoch + 1))
            metrics = validate_and_save(model, val_loader, args.num_classes, device,
                                        save_dir=pred_dir if args.save_pred_max > 0 else None,
                                        max_save=args.save_pred_max)
            print("Epoch {} val | mIoU {:.4f} | Dice {:.4f} | clDice {:.4f} | Acc {:.4f}".format(
                epoch + 1, metrics["miou"], metrics.get("dice_crack", 0),
                metrics.get("cldice_crack", 0), metrics["accuracy"]))
            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                save_ckpt(os.path.join(args.ckpt_dir, "best.pth"),
                          model, optimizer, epoch, global_step, best_miou)
                print("  -> 新的最佳 mIoU {:.4f}，已保存 best.pth".format(best_miou))

        if (epoch + 1) % args.save_epoch_freq == 0:
            save_ckpt(os.path.join(args.ckpt_dir, "ckpt_epoch_{:03d}.pth".format(epoch + 1)),
                      model, optimizer, epoch, global_step, best_miou)

    save_ckpt(os.path.join(args.ckpt_dir, "last.pth"),
              model, optimizer, args.epochs - 1, global_step, best_miou)
    plot_loss(epoch_losses, os.path.join(args.ckpt_dir, "loss_curve.png"))
    print("Training completed. best mIoU {:.4f}. Logs saved to {}".format(best_miou, args.ckpt_dir))


if __name__ == "__main__":
    main()
