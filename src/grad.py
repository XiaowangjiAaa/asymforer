import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# 1. 加载模型（根据您的实际模型路径调整）
from src.AsymFormer import B0_T


def load_model(model_path, device):
    """加载预训练的 AsymFormer 模型"""
    model = B0_T()  # 根据您的模型参数初始化
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


# 3. 创建模型包装器（处理双输入）
class AsymFormerWrapper(nn.Module):
    """包装 AsymFormer 模型以兼容 Grad-CAM"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        """
        前向传播
        输入: (rgb_tensor, depth_tensor)
        """
        rgb, depth = input_tensor
        return self.model(rgb, depth)


# 4. 主函数
def main():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "path/to/your/model.pth"  # 替换为您的模型路径
    rgb_path = "path/to/your/rgb_image.jpg"  # RGB图像路径
    depth_path = "path/to/your/depth_image.png"  # 深度图路径
    output_dir = "output"  # 输出目录
    target_size = (224, 224)  # 模型输入尺寸

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载模型
        model = load_model(model_path, device)
        wrapped_model = AsymFormerWrapper(model)

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

        # 组合输入
        net_input = (input_tensor, depth_tensor)

        # 选择目标层（根据您的模型结构调整）
        # 示例：选择最后一个卷积块
        target_layers = [model.encoder.blocks[-1]]

        # 初始化 Grad-CAM
        cam = GradCAM(model=wrapped_model,
                      target_layers=target_layers,
                      use_cuda=torch.cuda.is_available())

        # 生成热力图
        grayscale_cam = cam(net_input)
        grayscale_cam = grayscale_cam[0, :]  # 取第一个（也是唯一一个）结果

        # 可视化结果
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 保存结果
        base_name = os.path.basename(rgb_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_cam.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        print(f"Grad-CAM 结果已保存至: {output_path}")

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