import argparse
import dataclasses
import os
import sys

import torch

from src.AsymFormer_GSSA_ablation import AblationConfig, ABLATIONS, B0_T


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        print("[error] 缺少 pyyaml，请先执行: pip install pyyaml")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_config():
    return {f.name: f.default for f in dataclasses.fields(AblationConfig)}


def build_models(name, num_classes):
    cfg = AblationConfig(**ABLATIONS[name])
    return B0_T(num_classes, cfg), cfg


def check_one(exp, num_classes, input_size, device, defaults):
    name = exp["name"]
    H, W = input_size
    result = {"name": name, "group": exp.get("group", "?"), "ok": True, "msgs": []}

    # 1. 名称是否在注册表内
    if name not in ABLATIONS:
        result["ok"] = False
        result["msgs"].append("名称不在 ABLATIONS 注册表中")
        return result

    # 2. YAML 配置与代码注册表是否一致
    merged = dict(defaults)
    for k, v in exp.get("config", {}).items():
        if k not in merged:
            result["ok"] = False
            result["msgs"].append("未知配置项: {}".format(k))
            continue
        merged[k] = v
    if dict(defaults, **ABLATIONS[name]) != merged:
        result["ok"] = False
        result["msgs"].append("YAML 配置与 ABLATIONS[{}] 不一致".format(name))

    # 3. 构建模型 + 前向
    try:
        model, cfg = build_models(name, num_classes)
        model.eval().to(device)
        image = torch.rand(1, 3, H, W, device=device)
        depth = torch.rand(1, 1, H, W, device=device)
        with torch.no_grad():
            out = model(image, depth)
        exp_shape = (1, num_classes, H, W)
        if tuple(out.shape) != exp_shape:
            result["ok"] = False
            result["msgs"].append("输出形状 {} != 期望 {}".format(tuple(out.shape), exp_shape))
        else:
            result["out_shape"] = tuple(out.shape)
    except Exception as e:
        result["ok"] = False
        result["msgs"].append("构建/前向异常: {}".format(repr(e)))
        return result

    # 4. 统计 FLOPs / 参数量（thop 对 FFT 等算子可能不支持，失败则跳过）
    try:
        from thop import profile
        macs, params = profile(model, inputs=(image, depth,), verbose=False)
        result["flops"] = macs / 1e9
        result["params"] = params / 1e6
    except Exception:
        result["flops"] = float("nan")
        result["params"] = float("nan")

    return result


def main():
    parser = argparse.ArgumentParser(description="检查消融实验可行性")
    parser.add_argument("--config", type=str, default="./ablation_config.yaml")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--input-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    data = load_yaml(args.config)
    num_classes = args.num_classes if args.num_classes is not None else data.get("num_classes", 2)
    input_size = tuple(args.input_size) if args.input_size is not None else tuple(data.get("input_size", [480, 640]))
    experiments = data.get("experiments", [])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    defaults = default_config()
    print("device={}  num_classes={}  input_size={}  experiments={}".format(
        device, num_classes, input_size, len(experiments)))
    print("=" * 80)

    passed = failed = 0
    for exp in experiments:
        r = check_one(exp, num_classes, input_size, device, defaults)
        if r["ok"]:
            passed += 1
            flops = "{:.3f}G".format(r.get("flops", 0)) if r.get("flops") == r.get("flops") else "N/A"
            params = "{:.2f}M".format(r.get("params", 0)) if r.get("params") == r.get("params") else "N/A"
            print("[PASS] {:<16} G{}  out={}  FLOPs={}  Params={}".format(
                r["name"], r["group"], r.get("out_shape", "?"), flops, params))
        else:
            failed += 1
            print("[FAIL] {:<16} G{}  {}".format(r["name"], r["group"], "; ".join(r["msgs"])))

    print("=" * 80)
    print("通过 {}/{}，失败 {}".format(passed, len(experiments), failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
