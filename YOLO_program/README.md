# YOLO 项目

基于 Ultralytics YOLO 的目标检测项目。

## 环境配置

本项目使用 `yolo26` Conda 环境，包含以下核心依赖：

| 组件 | 版本 |
|------|------|
| Python | 3.12.13 |
| PyTorch | 2.7.1 (CUDA 11.8) |
| Ultralytics | 8.4.43 |
| OpenCV | 4.13.0.92 |

### 激活环境

```bash
# 方式一：手动激活
conda activate yolo26

# 方式二：在 VS Code 中打开终端（自动激活）
# 已配置 .vscode/settings.json，打开终端时会自动激活 yolo26 环境
```

### 运行项目

```bash
# 确保 yolo26 环境已激活
python main.py
```

## 项目结构

```
YOLO_program/
├── .vscode/
│   └── settings.json    # VS Code 配置（自动使用 yolo26 环境）
├── main.py              # 项目入口
├── requirements.txt     # 依赖清单
├── README.md            # 项目说明
└── .gitignore           # Git 忽略规则
```
