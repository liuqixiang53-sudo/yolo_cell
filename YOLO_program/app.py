"""
细菌显微图片识别与计数前端界面 (Gradio)
模型: YOLO26m best.pt (live_ecoli / dead_ecoli / live_saureus / dead_saureus)
用法: conda activate yolo26 && python app.py  →  http://localhost:7860
"""

import csv
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = (
    SCRIPT_DIR.parent / "runs" / "detect" / "runs" / "bacteria_detect" / "weights" / "best.pt"
)

CLASS_CN = {
    "live_ecoli": "活大肠杆菌",
    "dead_ecoli": "死大肠杆菌",
    "live_saureus": "活金葡菌",
    "dead_saureus": "死金葡菌",
}

CONF = 0.25
IOU = 0.45
DEVICE = "0" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"未找到模型权重: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


def warmup():
    global DEVICE
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        model.predict(dummy, conf=CONF, iou=IOU, device=DEVICE, verbose=False)
    except RuntimeError:
        if DEVICE != "cpu":
            DEVICE = "cpu"
            model.predict(dummy, conf=CONF, iou=IOU, device=DEVICE, verbose=False)
        else:
            raise


def detect(img_path):
    if img_path is None:
        raise gr.Error("请先上传一张显微图片")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise gr.Error("图片读取失败,请更换图片")

    t0 = time.time()
    results = model.predict(img_bgr, conf=CONF, iou=IOU, device=DEVICE, verbose=False)
    r = results[0]
    elapsed = time.time() - t0

    annotated_bgr = r.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    names = r.names
    boxes = r.boxes
    details = []
    counts = {}
    if boxes is not None and len(boxes):
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i]]
            name = names[cls_id]
            counts[name] = counts.get(name, 0) + 1
            details.append((i + 1, name, conf, x1, y1, x2, y2))

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    table = [[CLASS_CN.get(n, n), n, c] for n, c in ranked]
    table.append(["总计", "—", total])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(tempfile.mkdtemp(prefix="bacteria_"))
    img_out = out_dir / f"检测标注_{stamp}.png"
    csv_out = out_dir / f"计数结果_{stamp}.csv"
    cv2.imwrite(str(img_out), annotated_bgr)

    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "类别(中文)", "类别(英文)", "置信度", "x1", "y1", "x2", "y2"])
        for idx, name, conf, x1, y1, x2, y2 in details:
            w.writerow([idx, CLASS_CN.get(name, name), name, f"{conf:.3f}",
                        f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
        w.writerow([])
        w.writerow(["汇总", "类别(中文)", "类别(英文)", "数量"])
        for n, c in ranked:
            w.writerow(["", CLASS_CN.get(n, n), n, c])
        w.writerow(["", "总计", "—", total])

    device_name = "GPU" if DEVICE != "cpu" else "CPU"
    info = (f"共检出 {total} 个目标 | 图片 {img_bgr.shape[1]}×{img_bgr.shape[0]} | "
            f"推理 {elapsed * 1000:.0f} ms | {device_name} | conf≥{CONF}")

    return annotated_rgb, table, str(img_out), str(csv_out), info


with gr.Blocks(title="细菌显微图片识别与计数") as demo:
    gr.Markdown(
        "## 细菌显微图片识别与计数\n"
        "模型: **YOLO26m** (best.pt) | 类别: 活大肠杆菌 · 死大肠杆菌 · 活金葡菌 · 死金葡菌\n"
        "上传显微图片后自动识别,可下载标注图与计数 CSV。"
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="上传显微图片", type="filepath")
            run_btn = gr.Button("开始识别", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="识别结果")
            info_out = gr.Textbox(label="概要", interactive=False)
            table_out = gr.Dataframe(
                headers=["类别(中文)", "类别(英文)", "数量"],
                label="计数统计",
                interactive=False,
            )
            with gr.Row():
                file_img = gr.File(label="标注图下载")
                file_csv = gr.File(label="计数 CSV 下载")

    outputs = [img_out, table_out, file_img, file_csv, info_out]
    run_btn.click(detect, inputs=img_in, outputs=outputs)
    img_in.upload(detect, inputs=img_in, outputs=outputs)

if __name__ == "__main__":
    warmup()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
