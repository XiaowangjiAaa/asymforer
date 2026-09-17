import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from src.AsymFormer import B0_T, GSA_Module  # 导入 GSA_Module 类


# 1. 加载模型
def load_model(model_path, device):
    """加载预训练的 AsymFormer 模型"""
    model = B0_T(num_classes=2)
    print(model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    return model


# 2. 图像预处理函数
def preprocess_rgb(rgb_path, target_size=(224, 224)):
    """预处理 RGB 图像"""
    img = cv2.imread(rgb_path)
    if img is None:
        raise ValueError(f"无法加载 RGB 图像: {rgb_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = np.float32(img) / 255
    return img


def preprocess_depth(depth_path, target_size=(224, 224)):
    """预处理深度图"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"无法加载深度图: {depth_path}")

    # 确保深度图是单通道
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]

    depth = cv2.resize(depth, target_size)

    # 归一化深度图 (0-1范围)
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 65535.0
    elif depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.0
    else:
        depth = depth.astype(np.float32)
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    return depth


# 3. 自定义 Grad-CAM 函数（针对 GSA_Module 层）
def custom_grad_cam(model, target_layer, input_tensor, depth_tensor):
    """自定义 Grad-CAM 实现（适用于 GSA_Module 层）"""
    # 创建激活和梯度钩子
    activations = []
    gradients = []

    # 用于调试的输出信息
    debug_info = {"forward_output": None, "backward_output": None}

    def forward_hook(module, input, output):
        # 保存输出用于调试
        debug_info["forward_output"] = output

        # 尝试提取张量
        if isinstance(output, torch.Tensor):
            activations.append(output.detach())
        elif isinstance(output, tuple):
            # 尝试提取元组中的第一个张量
            for item in output:
                if isinstance(item, torch.Tensor):
                    activations.append(item.detach())
                    break
        else:
            print(f"警告: 无法提取激活张量, 输出类型: {type(output)}")

    def backward_hook(module, grad_input, grad_output):
        # 保存输出用于调试
        debug_info["backward_output"] = grad_output

        # 尝试提取梯度张量
        if grad_output is None:
            print("警告: 梯度输出为 None")
            return

        if isinstance(grad_output, tuple):
            # 尝试提取元组中的第一个张量
            for item in grad_output:
                if isinstance(item, torch.Tensor):
                    gradients.append(item.detach())
                    break
        elif isinstance(grad_output, torch.Tensor):
            gradients.append(grad_output.detach())
        else:
            print(f"警告: 无法提取梯度张量, 输出类型: {type(grad_output)}")

    # 注册钩子
    hook = target_layer.register_forward_hook(forward_hook)
    hook_grad = target_layer.register_backward_hook(backward_hook)

    # 前向传播
    output = model(input_tensor, depth_tensor)

    # 对于分割模型，我们通常对特定类别的特征图感兴趣
    # 这里我们假设我们关注的是类别1（裂缝）的激活
    target_class = 1
    one_hot = torch.zeros_like(output)
    one_hot[:, target_class, :, :] = 1

    # 反向传播
    model.zero_grad()
    output.backward(gradient=one_hot)

    # 检查是否成功获取激活和梯度
    if not activations:
        print("错误: 未能获取激活")
        print(f"前向输出类型: {type(debug_info['forward_output'])}")
        if isinstance(debug_info['forward_output'], tuple):
            print(f"前向输出元组长度: {len(debug_info['forward_output'])}")
            for i, item in enumerate(debug_info['forward_output']):
                print(f"前向输出[{i}]类型: {type(item)}")

    if not gradients:
        print("错误: 未能获取梯度")
        print(f"反向输出类型: {type(debug_info['backward_output'])}")
        if isinstance(debug_info['backward_output'], tuple):
            print(f"反向输出元组长度: {len(debug_info['backward_output'])}")
            for i, item in enumerate(debug_info['backward_output']):
                print(f"反向输出[{i}]类型: {type(item)}")

    if not activations or not gradients:
        raise RuntimeError("未能成功获取激活或梯度")

    # 获取激活和梯度
    act = activations[0]
    grad = gradients[0]

    # 打印激活和梯度的形状
    print(f"激活形状: {act.shape}")
    print(f"梯度形状: {grad.shape}")

    # 计算权重（全局平均池化梯度）
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)

    # 计算热力图
    cam = torch.sum(weights * act, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)

    # 移除钩子
    hook.remove()
    hook_grad.remove()

    return cam.squeeze().cpu().numpy()


# 4. 主函数
def main():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model_M1/asym_DF_ccffrepvgg_2/ckpt_epoch_120.00.pth"
    rgb_path = r"H:\xcy\AsymFormer-main\AsymFormer-main\RGB-Dcrackdataset\images\483_original.jpg"
    depth_path = r"H:\xcy\AsymFormer-main\AsymFormer-main\RGB-Dcrackdataset\depths\483_original.jpg"
    output_dir = r"H:\xcy\AsymFormer-main\AsymFormer-main"
    target_size = (640, 480)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载模型
        model = load_model(model_path, device)

        # 预处理图像
        rgb_img = preprocess_rgb(rgb_path, target_size)
        depth_img = preprocess_depth(depth_path, target_size)

        # 准备RGB张量
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(device)

        # 准备深度张量
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0)
        depth_tensor = depth_tensor.to(device)

        # 选择目标层 - GSA_Module
        target_layer = None

        # 遍历模型找到第一个 GSA_Module 实例
        for name, module in model.named_modules():
            if isinstance(module, GSA_Module):
                target_layer = module
                print(f"找到 GSA_Module 层: {name}")
                break

        if target_layer is None:
            raise RuntimeError("未找到 GSA_Module 层")

        print(f"选择的目标层: {target_layer}")

        # 使用自定义 Grad-CAM 函数
        grayscale_cam = custom_grad_cam(
            model=model,
            target_layer=target_layer,
            input_tensor=input_tensor,
            depth_tensor=depth_tensor
        )

        # 调整热力图尺寸以匹配原始图像
        grayscale_cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

        # 可视化结果
        visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)

        # 保存结果
        base_name = os.path.basename(rgb_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_gsa_module_cam.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        print(f"GSA_Module Grad-CAM 结果已保存至: {output_path}")

        # 可选：保存原始RGB和深度图用于对比
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_rgb.jpg"),
                    cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        depth_vis = (depth_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_depth.jpg"), depth_vis)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()