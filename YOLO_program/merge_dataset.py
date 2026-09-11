"""
合并新旧数据集脚本
====================
1. 汇总全部图片+标签:
   - 来源1: dataset/images/{train,val} + dataset/labels/{train,val}  (121 张, A 前缀)
   - 来源2: Unlable_2/*.png + labels_2/*.txt                         (92 张)
2. 删除 uncertain 类别 (class id = 4) 的所有标签行
3. 全量洗牌 (seed=42) 后按 80/20 重新划分 train/val
4. 重建 dataset/images 与 dataset/labels, 删除旧 cache
5. 输出 dataset/split_manifest.txt 记录本次划分名单

用法:
    cd /home/summer/桌面/AAAA_To_Jinjin_work/YOLO
    python YOLO_program/merge_dataset.py
"""
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
STAGE = DATASET / "_stage"

DROP_CLASS = 4          # uncertain
SEED = 42
VAL_RATIO = 0.2


def collect_pairs():
    """返回 [(stem, img_path, lbl_path)]"""
    pairs = []
    seen = set()
    # 来源1: 现有 dataset
    for split in ("train", "val"):
        lbl_dir = DATASET / "labels" / split
        img_dir = DATASET / "images" / split
        for lbl in sorted(lbl_dir.glob("*.txt")):
            if lbl.name.startswith("."):
                continue
            img = img_dir / (lbl.stem + ".png")
            if not img.exists():
                raise SystemExit(f"缺少图片: {img}")
            pairs.append((lbl.stem, img, lbl))
            seen.add(lbl.stem)
    # 来源2: 新数据
    for lbl in sorted((ROOT / "labels_2").glob("*.txt")):
        if lbl.name.startswith("."):
            continue
        if lbl.stem in seen:
            print(f"[去重] 跳过重复项: {lbl.stem} (已存在于 dataset, 内容一致)")
            continue
        img = ROOT / "Unlable_2" / (lbl.stem + ".png")
        if not img.exists():
            raise SystemExit(f"缺少图片: {img}")
        pairs.append((lbl.stem, img, lbl))
        seen.add(lbl.stem)
    return pairs


def read_filtered(lbl_path):
    """读取标签并删除 uncertain(class=4) 行"""
    rows = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cls = int(line.split()[0])
        if cls == DROP_CLASS:
            continue
        rows.append(line)
    return rows


def main():
    pairs = collect_pairs()
    print(f"共汇总 {len(pairs)} 组图片+标签")

    # 1) 暂存: 过滤标签 + 复制图片到 _stage（此时所有源文件都还在）
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    data = {}  # stem -> (stage_img_path, filtered_rows)
    kept = dropped = 0
    for stem, img, lbl in pairs:
        rows = read_filtered(lbl)
        dropped += sum(
            1 for l in lbl.read_text(encoding="utf-8").splitlines()
            if l.strip() and int(l.split()[0]) == DROP_CLASS
        )
        kept += len(rows)
        stage_img = STAGE / (stem + ".png")
        shutil.copy2(img, stage_img)
        data[stem] = (stage_img, rows)

    # 2) 洗牌 + 划分
    rng = random.Random(SEED)
    stems = list(data.keys())
    rng.shuffle(stems)
    n_val = round(len(stems) * VAL_RATIO)
    print(f"划分: train={len(stems) - n_val}, val={n_val}")
    assign = {s: ("val" if i < n_val else "train") for i, s in enumerate(stems)}

    # 3) 清空并重建 dataset
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = DATASET / sub / split
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
    for cache in DATASET.glob("labels/*.cache"):
        cache.unlink()

    # 4) 分发
    manifest_train = []
    manifest_val = []
    for stem in stems:
        split = assign[stem]
        stage_img, rows = data[stem]
        shutil.copy2(stage_img, DATASET / "images" / split / (stem + ".png"))
        (DATASET / "labels" / split / (stem + ".txt")).write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        (manifest_train if split == "train" else manifest_val).append(stem)

    # 5) 写清单
    manifest = [
        f"# 数据集划分清单 (seed={SEED}, val_ratio={VAL_RATIO})",
        "# 类别: 0=live_ecoli 1=dead_ecoli 2=live_saureus 3=dead_saureus (已删除 4=uncertain)",
        f"# train={len(manifest_train)}, val={len(manifest_val)}",
        "",
        "[train]",
    ] + [s + ".png" for s in sorted(manifest_train)] + ["", "[val]"] + [
        s + ".png" for s in sorted(manifest_val)
    ]
    (DATASET / "split_manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )

    # 6) 清理暂存
    shutil.rmtree(STAGE, ignore_errors=True)

    print(f"完成: 保留实例 {kept}, 丢弃 uncertain 实例 {dropped}")
    print("清单已写入 dataset/split_manifest.txt")


if __name__ == "__main__":
    main()