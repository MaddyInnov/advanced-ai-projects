#!/usr/bin/env python3
"""SkyTrack benchmarks — everything measured, nothing asserted.

1. Detection: YOLOv8n (COCO-pretrained) validated on COCO128 via the
   Ultralytics validator -> mAP50, mAP50-95, precision, recall.
2. Tracking: YOLOv8n detections + the from-scratch SORT tracker
   (src/sort_tracker.py) on the OpenCV sample surveillance video
   vtest.avi -> frames processed, unique confirmed tracks, per-class
   track counts, end-to-end FPS and per-frame latency percentiles.
   (No ground-truth identities exist for vtest.avi, so MOTA/IDF1 are
   NOT claimed; track fragmentation is reported as observed.)
3. Enhancement: CLAHE pipeline on 30 COCO128 images degraded with a
   labelled synthetic low-light transform -> PSNR/SSIM before vs after.
4. Detection latency: single-image CPU latency distribution (640 px).

Outputs: results.json + annotated frames + tracked_video.mp4 in outputs/.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sort_tracker import SortTracker          # noqa: E402
from enhance import enhance_frame, degrade_lowlight, psnr, ssim  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"


def get_video() -> Path:
    p = DATA / "vtest.avi"
    if not p.exists():
        print("downloading vtest.avi ...")
        urllib.request.urlretrieve(VIDEO_URL, p)
    return p


def main() -> None:
    from ultralytics import YOLO

    results: dict = {"model": "YOLOv8n (COCO pretrained, Ultralytics)", "device": "CPU"}
    model = YOLO("yolov8n.pt")

    # ---------- 1. Detection benchmark on COCO128 ----------
    print("== detection validation on COCO128 ==")
    metrics = model.val(data="coco128.yaml", imgsz=640, verbose=False)
    results["detection_coco128"] = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "note": "COCO128 = first 128 COCO train2017 images (Ultralytics mirror); small-sample estimate, not full COCO val2017.",
    }
    print(results["detection_coco128"])

    # ---------- 4 (early). Single-image latency ----------
    imgs = sorted((Path(metrics.save_dir).parent.parent / "datasets").rglob("*.jpg")) if False else []
    sample_imgs = []
    try:
        from ultralytics.utils import ASSETS  # noqa
    except Exception:
        pass
    # use coco128 images downloaded by the validator
    cand = list(Path.home().glob("**/coco128/images/train2017/*.jpg")) or \
        list((ROOT).rglob("coco128/images/train2017/*.jpg")) or \
        list(Path("datasets").rglob("*.jpg")) or \
        list(Path("/tmp").rglob("coco128/images/train2017/*.jpg"))
    if not cand:
        import ultralytics
        ds_root = Path(ultralytics.settings.get("datasets_dir", "datasets"))
        cand = list(ds_root.rglob("coco128/images/train2017/*.jpg"))
    sample_imgs = [str(p) for p in cand[:40]]
    results["coco128_images_found"] = len(sample_imgs)

    if sample_imgs:
        lat = []
        for p in sample_imgs[:30]:
            im = cv2.imread(p)
            t0 = time.perf_counter()
            model.predict(im, imgsz=640, verbose=False)
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat = np.array(lat[3:])  # drop warm-up
        results["detection_latency_cpu_ms"] = {
            "mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)), "n": int(len(lat)),
        }
        print("latency:", results["detection_latency_cpu_ms"])

    # ---------- 3. Enhancement benchmark ----------
    if sample_imgs:
        print("== enhancement benchmark ==")
        rows = []
        for i, p in enumerate(sample_imgs[:30]):
            im = cv2.imread(p)
            if im is None:
                continue
            deg = degrade_lowlight(im, seed=i)
            enh = enhance_frame(deg)
            rows.append((psnr(im, deg), psnr(im, enh), ssim(im, deg), ssim(im, enh)))
            if i == 0:
                cv2.imwrite(str(OUT / "enhance_original.jpg"), im)
                cv2.imwrite(str(OUT / "enhance_degraded.jpg"), deg)
                cv2.imwrite(str(OUT / "enhanced.jpg"), enh)
        arr = np.array(rows)
        results["enhancement_synthetic_lowlight"] = {
            "n_images": int(len(rows)),
            "psnr_before": float(arr[:, 0].mean()), "psnr_after": float(arr[:, 1].mean()),
            "ssim_before": float(arr[:, 2].mean()), "ssim_after": float(arr[:, 3].mean()),
            "note": "Degradation is synthetic (gamma 2.2 + Gaussian noise sigma 12) and labelled as such; enhancement recovery is measured against the clean originals.",
        }
        print(results["enhancement_synthetic_lowlight"])

    # ---------- 2. Tracking benchmark on vtest.avi ----------
    print("== tracking on vtest.avi ==")
    video = get_video()
    cap = cv2.VideoCapture(str(video))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUT / "tracked_video.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))
    tracker = SortTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    names = model.names
    track_ids: set[int] = set()
    track_classes: dict[int, list[int]] = {}
    frame_times = []
    n_frames = 0
    det_per_frame = []
    MAX_FRAMES = 450
    while n_frames < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append([x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])])
        dets = np.array(dets) if dets else np.empty((0, 6))
        det_per_frame.append(len(dets))
        tracks = tracker.update(dets)
        for tr in tracks:
            x1, y1, x2, y2, tid, cls, conf = tr
            tid = int(tid)
            track_ids.add(tid)
            track_classes.setdefault(tid, []).append(int(cls))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(frame, f"ID {tid} {names[int(cls)]}", (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        frame_times.append((time.perf_counter() - t0) * 1000.0)
        writer.write(frame)
        if n_frames in (30, 200, 400):
            cv2.imwrite(str(OUT / f"tracked_frame_{n_frames}.jpg"), frame)
        n_frames += 1
    cap.release()
    writer.release()
    ft = np.array(frame_times[5:])
    per_class_tracks: dict[str, int] = {}
    for tid, clss in track_classes.items():
        cls_name = names[int(np.bincount(clss).argmax())]
        per_class_tracks[cls_name] = per_class_tracks.get(cls_name, 0) + 1
    results["tracking_vtest"] = {
        "video": "vtest.avi (OpenCV sample surveillance video, elevated camera)",
        "frames_processed": int(n_frames),
        "resolution": f"{width}x{height}",
        "unique_confirmed_tracks": int(len(track_ids)),
        "tracks_by_class": per_class_tracks,
        "mean_detections_per_frame": float(np.mean(det_per_frame)) if det_per_frame else 0.0,
        "end_to_end_fps_cpu": float(1000.0 / ft.mean()) if len(ft) else 0.0,
        "frame_latency_ms": {"mean": float(ft.mean()), "p50": float(np.percentile(ft, 50)),
                             "p95": float(np.percentile(ft, 95))} if len(ft) else {},
        "note": "No identity ground truth for vtest.avi; MOTA/IDF1 are NOT claimed. Unique-track count includes fragmentation from occlusions — reported as observed.",
    }
    print(results["tracking_vtest"])

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
