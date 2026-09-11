"""
YOLO 细菌检测训练项目
识别 4 类细菌：D_SA (死金葡菌)、D_EC (死大肠杆菌)、L_EC (活大肠杆菌)、L_SA (活金葡菌)
使用 yolo26 环境 (Python 3.12, PyTorch 2.7.1, CUDA 11.8, Ultralytics 8.4.43)
"""

import sys
import torch
import ultralytics
from pathlib import Path
from ultralytics import YOLO

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()
# 构建 data.yaml 的绝对路径
DATA_YAML = SCRIPT_DIR.parent / "dataset" / "data.yaml"
# 训练输出的最佳模型权重（绝对路径, 4 类）
BEST_PT = SCRIPT_DIR.parent / "runs" / "detect" / "runs" / "bacteria_detect" / "weights" / "best.pt"


def check_environment():
    """检查当前运行环境"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"Ultralytics 版本: {ultralytics.__version__}")
    print("=" * 50)


def train_model():
    """训练 YOLO26 细菌检测模型"""
    print("\n" + "=" * 50)
    print("开始训练 YOLO26 细菌检测模型")
    print("=" * 50)
    
    # 加载预训练模型（绝对路径, 位于项目根目录）
    model = YOLO(str(SCRIPT_DIR.parent / "yolo26m.pt"))
    
    # 训练配置
    results = model.train(
        data=str(DATA_YAML),  # 数据集配置
        epochs=100,                      # 训练轮数
        imgsz=640,                       # 图像尺寸
        batch=8,                         # 批量大小（小数据集）
        patience=20,                     # 早停耐心值
        save=True,                       # 保存模型
        project=str(SCRIPT_DIR.parent / "runs" / "detect" / "runs"),   # 输出目录 YOLO/runs/detect/runs
        name="bacteria_detect",          # 实验名称
        exist_ok=True,                   # 覆盖已有实验
        pretrained=True,                 # 使用预训练权重
        optimizer="AdamW",               # 优化器
        lr0=0.001,                       # 初始学习率
        lrf=0.01,                        # 最终学习率
        momentum=0.937,                  # 动量
        weight_decay=0.0005,              # 权重衰减
        warmup_epochs=3.0,               # 预热轮数
        warmup_momentum=0.8,             # 预热动量
        warmup_bias_lr=0.1,              # 预热偏置学习率
        box=7.5,                         # 边框损失权重
        cls=0.5,                         # 分类损失权重
        dfl=1.5,                         # DFL 损失权重
        hsv_h=0.015,                     # 色相增强
        hsv_s=0.7,                       # 饱和度增强
        hsv_v=0.4,                       # 明度增强
        degrees=0.0,                      # 旋转增强
        translate=0.1,                    # 平移增强
        scale=0.5,                        # 缩放增强
        shear=0.0,                        # 剪切增强
        perspective=0.0,                  # 透视增强
        flipud=0.0,                      # 上下翻转
        fliplr=0.5,                      # 左右翻转
        mosaic=1.0,                       # 马赛克增强
        mixup=0.1,                       # MixUp 增强
        copy_paste=0.0,                   # Copy-Paste 增强
        verbose=True,                     # 详细输出
        device=0 if torch.cuda.is_available() else "cpu"  # 使用 GPU
    )
    
    print("\n训练完成!")
    print(f"最佳模型路径: {BEST_PT}")
    
    return results


def validate_model():
    """验证训练好的模型"""
    print("\n" + "=" * 50)
    print("验证模型性能")
    print("=" * 50)
    
    # 加载训练好的模型
    model = YOLO(str(BEST_PT))
    
    # 在验证集上评估
    results = model.val(
        data=str(DATA_YAML),
        split="val"
    )
    
    print(f"\n验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    
    return results


def main():
    """主函数"""
    check_environment()
    
    print("\n" + "=" * 50)
    print("YOLO26 细菌检测模型")
    print("=" * 50)
    print("类别说明:")
    print("  0: live_ecoli   - 活大肠杆菌")
    print("  1: dead_ecoli   - 死大肠杆菌")
    print("  2: live_saureus - 活金黄色葡萄球菌")
    print("  3: dead_saureus - 死金黄色葡萄球菌")
    print("=" * 50)
    
    # 训练模型
    train_model()
    
    # 验证模型
    validate_model()
    
    print("\n✅ 训练和验证完成!")


if __name__ == "__main__":
    main()