import argparse
import numpy as np
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torchvision
import time
from torch.utils.data import DataLoader
import datetime
import cv2
from collections import OrderedDict
import torch.optim
import NYUv2_dataloader as Data
from src.AsymFormer import B0_T

#from src.Unet import B0_T
from utils import utils
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc

pth_dir = './model_M1/asym_GSACCFF_5e-6_1e-4_10/ckpt_epoch_200.00.pth'
model = B0_T(num_classes=2)

parser = argparse.ArgumentParser(description='RGBD SementicSegmentation')
parser.add_argument('--data-dir', default='./RGB-Dcrackdataset', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output', default='./result/crackpredict', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='./model/non_local_5173.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=2, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if output image')

args = parser.parse_args()

image_w = 640
image_h = 480
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]


def set_seed(seed=2333):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_block_pretrain_weight(model, pretrain_path):
    model_dict = model.state_dict()
    pretrain_dict = torch.load(pretrain_path)['state_dict']
    new_state_dict = OrderedDict()
    new_state_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}

    model.load_state_dict(new_state_dict, strict=False)


# transform
class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        label = label.astype(np.int16)
        # Bi-linear
        image = cv2.resize(image, (image_w, image_h), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (image_w, image_h), cv2.INTER_NEAREST)
        label = cv2.resize(label, (image_w, image_h), cv2.INTER_NEAREST)

        return {'image': image, 'depth': depth, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float64)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float()}


class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        image = image / 255
        depth = depth / 1000

        image = torchvision.transforms.Normalize(
            mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
            std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)

        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])(depth)
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth

        return sample


def visualize_result(img, depth, label, preds, info, args):
    img = img.squeeze(0).transpose(0, 2, 1)
    dep = depth.squeeze(0).squeeze(0)
    dep = (dep * 255 / dep.max()).astype(np.uint8)
    dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    dep = dep.transpose(2, 1, 0)
    seg_color = utils.color_label_eval(label)
    pred_color = utils.color_label_eval(preds)

    im_vis = np.concatenate((img, dep, seg_color, pred_color),
                            axis=1).astype(np.uint8)
    im_vis = im_vis.transpose(2, 1, 0)

    img_name = str(info)
    cv2.imwrite(os.path.join(args.output,
                             img_name + '.png'), im_vis)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def inference():
    device = torch.device("cuda:0")
    _load_block_pretrain_weight(model, pth_dir)
    model.eval()
    model.to(device)

    test_txt_path = os.path.join(args.data_dir, 'test.txt')
    with open(test_txt_path, 'r') as f:
        lines = f.readlines()
    filenames = []
    for line in lines:
        parts = line.strip().split()
        image_path = parts[0]
        filename = os.path.basename(image_path)
        filenames.append(filename)

    val_data = Data.RGBD_Dataset(transform=torchvision.transforms.Compose([scaleNorm(),
                                                                           ToTensor(),
                                                                           Normalize()]),
                                 phase_train=False,
                                 data_dir=args.data_dir,
                                 txt_name='test.txt'
                                 )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化评估指标meter
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()  # TP (True Positive)
    b_meter = AverageMeter()  # 真实正例数 (Actual Positives)
    c_meter = AverageMeter()  # 预测正例数 (Predicted Positives)
    recall_meter = AverageMeter()  # 召回率
    f1_meter = AverageMeter()  # F1分数

    t = 0
    acc_collect = []
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((len(val_loader), 1))
    dummy_rgb = torch.rand([1, 3, 480, 640], device=device)
    dummy_depth = torch.rand([1, 1, 480, 640], device=device)
    args.pred_save_dir = os.path.join(args.output, 'predictions')
    os.makedirs(args.pred_save_dir, exist_ok=True)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_rgb, dummy_depth)

        for batch_idx, sample in enumerate(val_loader):
            origin_image = sample['origin_image'].numpy()
            origin_depth = sample['origin_depth'].numpy()
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].numpy()
            unique, counts = np.unique(label, return_counts=True)
            print(f"标签分布：{dict(zip(unique, counts))}")
            name = filenames[batch_idx]
            starter.record()
            pred = model(image, depth)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[batch_idx] = curr_time

            output = torch.max(pred, 1)[1]
            output = output.squeeze(0).cpu().numpy()

            acc, pix = accuracy(output, label)
            acc_collect.append(acc)
            intersection, union = intersectionAndUnion(output, label, args.num_class)
            acc_meter.update(acc, pix)
            a_m, b_m, c_m = macc(output, label, args.num_class)

            # 计算召回率和F1分数
            precision = a_m / (c_m + 1e-10)  # 精确率 = TP / (TP + FP)
            recall = a_m / (b_m + 1e-10)  # 召回率 = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)  # F1分数

            # 更新meter
            intersection_meter.update(intersection)
            union_meter.update(union)
            a_meter.update(a_m)
            b_meter.update(b_m)
            c_meter.update(c_m)
            recall_meter.update(recall)
            f1_meter.update(f1)

            output = torch.argmax(pred, dim=1)
            pred_mask = output.squeeze().cpu().numpy().astype(np.uint8)
            resized_mask = cv2.resize(pred_mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join(args.pred_save_dir, f'pred_{name}.png'),
                resized_mask
            )
            print('[{}] iter {}, accuracy: {:.4f}, recall: {:.4f}, f1: {:.4f}'
                  .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          batch_idx, acc, np.mean(recall), np.mean(f1)))

            if args.visualize:
                visualize_result(origin_image, origin_depth, label - 1, output - 1, batch_idx, args)

    # 计算每个类别的IoU
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    # 计算宏平均召回率和F1分数
    total_a = a_meter.sum
    total_b = b_meter.sum
    total_c = c_meter.sum

    precision_per_class = total_a / (total_c + 1e-10)
    recall_per_class = total_a / (total_b + 1e-10)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-10)

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'.format(
        iou.mean(), acc_meter.average() * 100))
    print('Recall per class:')
    for i, recall_val in enumerate(recall_per_class):
        print('  Class {}: {:.4f}'.format(i, recall_val))
    print('F1 Score per class:')
    for i, f1_val in enumerate(f1_per_class):
        print('  Class {}: {:.4f}'.format(i, f1_val))
    print('Macro Recall: {:.4f}, Macro F1: {:.4f}'.format(
        np.mean(recall_per_class), np.mean(f1_per_class)))

    print('平均推理时间：{:.2f}ms'.format(timings.sum() / len(val_loader)))
    np.save('SCC_SRM5', np.array(acc_collect))
    print(f'预测结果已保存至：{args.pred_save_dir}')


if __name__ == '__main__':
    set_seed()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    inference()