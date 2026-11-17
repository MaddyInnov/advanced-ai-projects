# SkyTrack — Real-Time UAV & Surveillance Video Analytics

Object detection, multi-object tracking and image enhancement for
UAV-based and elevated camera systems — the Computer Vision stack:
**YOLOv8 detection → SORT multi-object tracking (Kalman filter +
Hungarian assignment, implemented from scratch) → OpenCV enhancement
and real-time video analytics.**

Built to close the Computer Vision gap (object detection / recognition /
tracking, OpenCV, YOLO, real-time video analytics, multi-object
tracking, UAV camera analytics) with *measured* results on real data —
no synthetic headlines.

## Pipeline

```
video / image stream
   │
   ├─ (optional) enhancement: CLAHE + fast-NLM denoise + unsharp mask  [src/enhance.py]
   ▼
YOLOv8n detection (Ultralytics, COCO-pretrained, ONNX-exportable)
   ▼
SORT tracker — per-track Kalman filter (constant-velocity model),
IoU cost + Hungarian association, tentative/confirmed track lifecycle  [src/sort_tracker.py]
   ▼
annotated video + per-track analytics (IDs, classes, counts, FPS)
```

## Benchmarks (all measured — see `outputs/results.json`)

| Benchmark | Data | What is reported |
|---|---|---|
| Detection | COCO128 (real COCO images) | mAP@0.5, mAP@0.5:0.95, precision, recall |
| Detection latency | COCO128 images, CPU | mean / p50 / p95 ms per image @ 640 px |
| Tracking | vtest.avi (OpenCV sample surveillance video) | frames, unique confirmed tracks, tracks by class, end-to-end FPS, frame latency |
| Enhancement | LOLv2-Real test (100 real paired low/normal-light photographs) | PSNR / SSIM of raw and enhanced vs the real reference photos |
| Detection in real darkness | ExDark test (734 real low-light images, human annotations) | mAP raw vs CLAHE-enhanced |

Honesty notes:
- **All evaluations are on real data.** An earlier synthetic-degradation
  enhancement test was replaced by the LOLv2-Real paired evaluation and
  its numbers are no longer reported as evidence.
- COCO128 is a 128-image subset — a small-sample estimate, not full
  COCO val2017.
- vtest.avi has no identity ground truth, so **MOTA / IDF1 are not
  claimed**; track counts are reported as observed (fragmentation from
  occlusion included).
- **Negative result, reported in full:** CLAHE enhancement improves
  reference fidelity on LOLv2-Real (SSIM 0.169 → 0.400) but *reduces*
  detector AP on ExDark (mAP@0.5 0.485 → 0.374). Enhancement tuned for
  human perception is not automatically enhancement for machine
  perception; in the deployed configuration the detector eats raw
  frames and enhancement serves human viewing.

## Run

```bash
pip install ultralytics opencv-python-headless scipy
python src/run_benchmarks.py     # writes outputs/results.json, tracked_video.mp4, frames
```

## Files

- `src/sort_tracker.py` — SORT + Kalman filter from scratch (numpy/scipy only)
- `src/enhance.py` — CLAHE enhancement + PSNR/SSIM + labelled degradation
- `src/run_benchmarks.py` — the full measured benchmark suite
- `outputs/` — results.json, tracked video, annotated frames, enhancement triplet
- `report/` — detailed project report (DOCX + PDF)

## Known limitations / next steps

- YOLOv8n is COCO-pretrained, not fine-tuned on aerial data; a VisDrone
  fine-tune is the obvious accuracy lever for true UAV footage.
- SORT is appearance-free (motion + IoU only); occlusions fragment
  tracks. ByteTrack-style low-confidence association or a ReID embedding
  (DeepSORT) is the documented upgrade path.
- CPU-only evaluation; TensorRT / OpenVINO export is the deployment
  path for edge/Jetson targets.
