import os
import argparse

import torch

from train_gssa import (setup_seed, build_criterion, build_lr_scheduler,
                        build_train_loader, build_val_loader, train_one_epoch,
                        validate_and_save, save_ckpt, load_ckpt, plot_loss, make_scaler)
from src.AsymFormer_GSSA_ablation import build_model, ABLATIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation training for AsymFormer-GSSA")
    parser.add_argument("--ablation", type=str, required=True, choices=list(ABLATIONS))
    parser.add_argument("--data-dir", type=str, default="Xie's dataset")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--loss", type=str, default="ce_dice", choices=["ce", "ce_dice", "focal"])
    parser.add_argument("--ignore-index", type=int, default=-1)
    parser.add_argument("--crack-weight", type=float, default=0.7)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="default: ./model_M1/ablation/<name>")
    parser.add_argument("--save-epoch-freq", type=int, default=5)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-pred-max", type=int, default=100)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join("./model_M1/ablation", args.ablation)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = build_model(args.ablation, args.num_classes)
    print("[config] {} -> {}".format(args.ablation, cfg))

    train_loader = build_train_loader(args)
    val_loader = build_val_loader(args) if args.eval_every > 0 else None
    criterion = build_criterion(args).to(device)
    model = model.to(device)
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
