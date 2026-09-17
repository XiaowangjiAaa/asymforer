import argparse
import csv
import json
import os
import sys
import time

import torch

from train_gssa import (setup_seed, build_criterion, build_lr_scheduler,
                        build_train_loader, build_val_loader, train_one_epoch,
                        save_ckpt, load_ckpt, make_scaler, validate_and_save)
from src.AsymFormer_GSSA_ablation import build_model
from metrics import SegMetrics


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        print("[error] 缺少 pyyaml，请先执行: pip install pyyaml")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="批量队列训练 + 评估消融实验")
    p.add_argument("--config", type=str, default="./ablation_config.yaml")
    p.add_argument("--group", type=str, default="all",
                   help="all 或逗号分隔，如 '1' / '1,2' / '2,3,4'")
    p.add_argument("--data-dir", type=str, default="Xie's dataset")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--output-dir", type=str, default="./model_M1/ablation")
    # 训练超参
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--loss", type=str, default="ce_dice", choices=["ce", "ce_dice", "focal"])
    p.add_argument("--ignore-index", type=int, default=-1,
                   help="-1 表示不忽略背景（推荐，避免全预测裂缝的退化解）")
    p.add_argument("--crack-weight", type=float, default=0.7)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=2333)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--clip-grad-norm", type=float, default=None)
    p.add_argument("--print-freq", type=int, default=50)
    # 评估 / 保存
    p.add_argument("--eval-every", type=int, default=5,
                   help="每 N 个 epoch 验证一次：算指标 + 存预测图 + 更新 best.pth（0 关闭）")
    p.add_argument("--save-pred-max", type=int, default=100,
                   help="每次验证最多保存多少张预测可视化图（0 表示不保存）")
    p.add_argument("--save-freq", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--resume", action="store_true",
                   help="若该实验已存在 last.pth，则跳过训练直接评估")
    return p.parse_args()


def select_experiments(data, group_arg):
    exps = data.get("experiments", [])
    if group_arg == "all":
        return exps
    wanted = set(int(g) for g in group_arg.split(",") if g.strip())
    return [e for e in exps if int(e.get("group", 0)) in wanted]


def evaluate_full(model, loader, num_classes, device):
    model.eval()
    m = SegMetrics(num_classes, crack_class=1)
    with torch.no_grad():
        for sample in loader:
            image = sample["image"].to(device)
            depth = sample["depth"].to(device)
            label = sample["label"].numpy()
            pred = model(image, depth)
            pred_np = torch.argmax(pred, 1).cpu().numpy().squeeze(0).astype("int64")
            label_np = label.squeeze(0).astype("int64")
            m.update(pred_np, label_np)
    return m.summary()


def run_experiment(exp, args, device):
    name = exp["name"]
    group = exp.get("group", "?")
    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    last_path = os.path.join(out_dir, "last.pth")
    best_path = os.path.join(out_dir, "best.pth")
    metrics_path = os.path.join(out_dir, "metrics.json")

    if args.resume and os.path.exists(last_path):
        print("  [resume] {} 已存在，跳过训练".format(name))
    else:
        setup_seed(args.seed)
        model, cfg = build_model(name, args.num_classes)
        model.to(device)
        criterion = build_criterion(args).to(device)
        train_loader = build_train_loader(args)
        val_loader = build_val_loader(args)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, len(train_loader), args)
        scaler = make_scaler(args.amp)

        best_miou = 0.0
        global_step = 0
        t0 = time.time()
        for epoch in range(args.epochs):
            avg_loss, global_step, _elapsed = train_one_epoch(model, train_loader, criterion, optimizer,
                                                              scheduler, scaler, device, epoch, args, global_step)
            if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
                pred_dir = os.path.join(out_dir, "predictions", "epoch_{:03d}".format(epoch + 1))
                metrics = validate_and_save(model, val_loader, args.num_classes, device,
                                            save_dir=pred_dir if args.save_pred_max > 0 else None,
                                            max_save=args.save_pred_max)
                print("    Epoch {} val | mIoU {:.4f} | Dice {:.4f} | clDice {:.4f}".format(
                    epoch + 1, metrics["miou"], metrics.get("dice_crack", 0),
                    metrics.get("cldice_crack", 0)))
                if metrics["miou"] > best_miou:
                    best_miou = metrics["miou"]
                    save_ckpt(best_path, model, optimizer, epoch, global_step, best_miou)
                    print("      -> 新最佳 mIoU {:.4f}，已保存 best.pth".format(best_miou))
            if (epoch + 1) % args.save_freq == 0:
                save_ckpt(os.path.join(out_dir, "ckpt_epoch_{:03d}.pth".format(epoch + 1)),
                          model, optimizer, epoch, global_step, best_miou)
        save_ckpt(last_path, model, optimizer, args.epochs - 1, global_step, best_miou)
        print("  [train] {} 完成，耗时 {:.1f}s，最佳 mIoU {:.4f}".format(
            name, time.time() - t0, best_miou))

    # 加载最终模型并评估
    model, _ = build_model(name, args.num_classes)
    ckpt = best_path if os.path.exists(best_path) else last_path
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device)

    val_loader = build_val_loader(args)
    metrics = evaluate_full(model, val_loader, args.num_classes, device)
    metrics["name"] = name
    metrics["group"] = group
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


SUMMARY_FIELDS = ["name", "group", "miou", "accuracy", "precision", "recall",
                  "f1", "dice_crack", "iou_crack", "cldice_crack"]


def write_summary(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
    print("  [summary] 已写入 {}".format(path))


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_yaml(args.config)
    if args.num_classes is None:
        args.num_classes = data.get("num_classes", 2)
    exps = select_experiments(data, args.group)

    print("device={}  num_classes={}  训练队列（{} 个实验）:".format(device, args.num_classes, len(exps)))
    for i, e in enumerate(exps):
        print("  {:2d}. [G{}] {}".format(i + 1, e.get("group"), e["name"]))
    print("=" * 80)

    all_rows = []
    total = len(exps)
    run_start = time.time()
    for idx, e in enumerate(exps):
        print("\n>>> [G{}] {}  ({}/{})".format(e.get("group"), e["name"], idx + 1, total))
        exp_start = time.time()
        metrics = run_experiment(e, args, device)
        exp_elapsed = time.time() - exp_start
        done = idx + 1
        avg_per_exp = (time.time() - run_start) / done
        eta = avg_per_exp * (total - done)
        print("    mIoU {:.4f} | Acc {:.4f} | F1 {:.4f} | Dice {:.4f} | clDice {:.4f}".format(
            metrics["miou"], metrics["accuracy"], metrics["f1"],
            metrics.get("dice_crack", 0), metrics.get("cldice_crack", 0)))
        print("    [进度 {}/{}] 本实验用时 {:.1f}s | 整体已用时 {:.1f}s | 剩余约 {:.1f}s".format(
            done, total, exp_elapsed, time.time() - run_start, eta))
        all_rows.append(metrics)

    # 每个 group 一份 summary + 一份总 summary
    groups = sorted({r["group"] for r in all_rows})
    for g in groups:
        write_summary(os.path.join(args.output_dir, "summary_group{}.csv".format(g)),
                      [r for r in all_rows if r["group"] == g])
    write_summary(os.path.join(args.output_dir, "summary_all.csv"), all_rows)
    print("\n全部完成，总耗时 {:.1f}s（{:.2f}min）。".format(time.time() - run_start, (time.time() - run_start) / 60))


if __name__ == "__main__":
    main()
