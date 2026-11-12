#!/bin/bash
cd ~/workspace/advanced-ai-projects/densehar
echo "=== HARTH v2 (class-weight) launch $(date -Is) pid $$ ===" >> harth_v2.log
exec python3 train_real.py exp --dataset harth --name harth_p2_aug_c12_refine_bh10_cw --epochs 20 --seed 0 --classes 12 --augment --refine --boundary-head 1.0 --class-weight --no-test >> harth_v2.log 2>&1
