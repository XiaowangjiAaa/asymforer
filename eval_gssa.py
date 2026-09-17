import os
import argparse
import datetime

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2

import NYUv2_dataloader as Data
from src.AsymFormer_GSSA_ablation import build_model, ABLATIONS
from utils.utils import intersectionAndUnion, accuracy, AverageMeter, macc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AsymFormer-GSSA ablation")
    parser.add_argument("--ablation", type=str, default="full", choices=list(ABLATIONS))
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, default="./RGB-Dcrackdataset")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--output", type=str, default=None,
                        help="dir to save predicted masks (optional)")
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def load_model(name, num_classes, ckpt, device):
    model, cfg = build_model(name, num_classes)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.to(device)
    return model, cfg


def build_val_loader(data_dir):
    transform = T.Compose([Data.scaleNorm(), Data.ToTensor(), Data.Normalize()])
    dataset = Data.RGBD_Dataset(transform=transform, phase_train=False,
                                data_dir=data_dir, txt_name="test.txt")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.ablation, args.num_classes, args.ckpt, device)
    print("[model] {} | cfg={}".format(args.ablation, cfg))

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    loader = build_val_loader(args.data_dir)

    acc_meter = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()
    c_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    for i, sample in enumerate(loader):
        image = sample["image"].to(device)
        depth = sample["depth"].to(device)
        label = sample["label"].numpy()

        pred = model(image, depth)
        output = torch.argmax(pred, 1).squeeze(0).cpu().numpy()

        acc, pix = accuracy(output, label)
        inter, union = intersectionAndUnion(output, label, args.num_classes)
        a_m, b_m, c_m = macc(output, label, args.num_classes)

        precision = a_m / (c_m + 1e-10)
        recall = a_m / (b_m + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        acc_meter.update(acc, pix)
        inter_meter.update(inter)
        union_meter.update(union)
        a_meter.update(a_m)
        b_meter.update(b_m)
        c_meter.update(c_m)
        recall_meter.update(recall)
        f1_meter.update(f1)

        if args.output:
            mask = (output.astype(np.uint8))
            cv2.imwrite(os.path.join(args.output, "{:05d}.png".format(i)), mask)

        if (i + 1) % 50 == 0:
            print("[{}] {}/{} acc {:.4f}".format(
                datetime.datetime.now().strftime("%H:%M:%S"), i + 1, len(loader), acc))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    total_a = a_meter.sum
    total_b = b_meter.sum
    total_c = c_meter.sum
    precision = total_a / (total_c + 1e-10)
    recall = total_a / (total_b + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    print("\n[Eval Summary] ablation={} ckpt={}".format(args.ablation, args.ckpt))
    print("mIoU {:.4f} | Accuracy {:.4f} | Macro Recall {:.4f} | Macro F1 {:.4f}".format(
        iou.mean(), acc_meter.average(), recall.mean(), f1.mean()))
    for i, _iou in enumerate(iou):
        print("  class {:2d}: IoU {:.4f}  Recall {:.4f}  F1 {:.4f}".format(
            i, _iou, recall[i], f1[i]))


if __name__ == "__main__":
    main()
