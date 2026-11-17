#!/usr/bin/env python3
"""SkyTrack FULLY-REAL benchmarks — replaces the synthetic enhancement test.

1. Enhancement on REAL paired data: LOLv2-Real test split (100 pairs).
   Each pair is two real photographs of the same scene: a real low-light
   exposure (Input) and a real normal-light reference (GT), captured by
   changing exposure time / ISO (Wei et al., LOL dataset protocol).
   -> PSNR / SSIM of raw low-light vs GT, and of CLAHE-enhanced vs GT.

2. Detection on REAL low-light data: ExDark test split (735 real
   low-light images, human bounding-box annotations, 12 classes).
   ExDark labels are remapped to COCO class ids so the COCO-pretrained
   YOLOv8n can be scored honestly. Detection AP is measured twice:
   on the raw low-light images and on CLAHE-enhanced copies
   (enhance_frame_fast: CLAHE + unsharp, no NLM — the throughput
   variant, labelled as such).
   -> mAP50 / mAP50-95 raw vs enhanced: does enhancement help a real
      detector on real darkness? (Reported either way.)

Merges results into outputs/results.json under
'enhancement_real_lolv2' and 'detection_exdark_lowlight'.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from enhance import enhance_frame, enhance_frame_fast, psnr, ssim  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ExDark (12) -> COCO (80) class-id mapping
EXDARK_TO_COCO = {0: 1,   # Bicycle -> bicycle
                  1: 8,   # Boat -> boat
                  2: 39,  # Bottle -> bottle
                  3: 5,   # Bus -> bus
                  4: 2,   # Car -> car
                  5: 15,  # Cat -> cat
                  6: 56,  # Chair -> chair
                  7: 41,  # Cup -> cup
                  8: 16,  # Dog -> dog
                  9: 3,   # Motorbike -> motorcycle
                  10: 0,  # People -> person
                  11: 60}  # Table -> dining table
COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
              'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
              'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
              'hair drier', 'toothbrush']


def lol_enhancement() -> dict:
    inp_dir = DATA / "lolv2_hf" / "Test" / "Input"
    gt_dir = DATA / "lolv2_hf" / "Test" / "GT"
    pairs = sorted(inp_dir.glob("*.png"))
    print(f"LOLv2-Real pairs: {len(pairs)}")
    rows = []
    for i, p in enumerate(pairs):
        low = cv2.imread(str(p))
        gt = cv2.imread(str(gt_dir / p.name))
        if low is None or gt is None:
            continue
        if low.shape != gt.shape:
            gt = cv2.resize(gt, (low.shape[1], low.shape[0]))
        enh = enhance_frame(low)
        rows.append((psnr(gt, low), psnr(gt, enh), ssim(gt, low), ssim(gt, enh)))
        if i == 0:
            cv2.imwrite(str(OUT / "lol_low.jpg"), low)
            cv2.imwrite(str(OUT / "lol_enhanced.jpg"), enh)
            cv2.imwrite(str(OUT / "lol_gt.jpg"), gt)
    arr = np.array(rows)
    res = {"dataset": "LOLv2-Real test split (real paired photographs, exposure-varied capture)",
           "n_pairs": int(len(rows)),
           "psnr_raw_vs_gt": float(arr[:, 0].mean()), "psnr_enhanced_vs_gt": float(arr[:, 1].mean()),
           "ssim_raw_vs_gt": float(arr[:, 2].mean()), "ssim_enhanced_vs_gt": float(arr[:, 3].mean()),
           "note": "Both the low-light inputs and the normal-light references are real photographs; nothing in this evaluation is synthetic."}
    print(res)
    return res


def prepare_exdark_variant(name: str, transform) -> Path:
    """Build a YOLO dataset folder for one variant (raw / enhanced)."""
    src_img = DATA / "exdark_hf" / "data" / "test" / "images"
    src_lab = DATA / "exdark_hf" / "data" / "test" / "labels"
    root = DATA / f"exdark_{name}"
    img_out = root / "images"
    lab_out = root / "labels"
    if img_out.exists():
        shutil.rmtree(root)
    img_out.mkdir(parents=True)
    lab_out.mkdir(parents=True)
    n = 0
    for img_path in sorted(src_img.glob("*.jpg")):
        lab_path = src_lab / (img_path.stem + ".txt")
        if not lab_path.exists():
            continue
        # remap labels ExDark -> COCO
        lines = []
        for line in lab_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = EXDARK_TO_COCO.get(int(float(parts[0])))
            if cls is None:
                continue
            lines.append(" ".join([str(cls)] + parts[1:]))
        (lab_out / (img_path.stem + ".txt")).write_text("\n".join(lines))
        im = cv2.imread(str(img_path))
        if transform is not None:
            im = transform(im)
        cv2.imwrite(str(img_out / img_path.name), im)
        n += 1
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images\nval: images\nnames:\n" +
        "\n".join(f"  {i}: {nm}" for i, nm in enumerate(COCO_NAMES)) + "\n")
    print(f"variant {name}: {n} images")
    return yaml_path


def exdark_detection() -> dict:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    out = {"dataset": "ExDark test split (734 real low-light images, human annotations; labels remapped ExDark->COCO)",
           "enhancement_variant": "enhance_frame_fast (CLAHE + unsharp, no NLM) for the enhanced copy"}
    for name, transform in [("raw", None), ("enhanced", enhance_frame_fast)]:
        yaml_path = prepare_exdark_variant(name, transform)
        print(f"== validating {name} ==")
        m = model.val(data=str(yaml_path), imgsz=640, verbose=False,
                      project=str(OUT / "exdark_val"), name=name, exist_ok=True)
        out[name] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                     "precision_mean": float(m.box.mp), "recall_mean": float(m.box.mr)}
        print(name, out[name])
    return out


def main() -> None:
    results = {}
    if (OUT / "results.json").exists():
        results = json.loads((OUT / "results.json").read_text())
    results["enhancement_real_lolv2"] = lol_enhancement()
    results["detection_exdark_lowlight"] = exdark_detection()
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("merged into", OUT / "results.json")


if __name__ == "__main__":
    main()
