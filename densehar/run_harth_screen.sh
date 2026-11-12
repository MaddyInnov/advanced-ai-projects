#!/bin/bash
cd ~/workspace/advanced-ai-projects/densehar
echo "=== HARTH screen launch $(date -Is) pid $$ ===" >> harth_bh10.log
exec python3 train_real.py exp --dataset harth --name harth_p2_aug_c12_refine_bh10 --epochs 20 --seed 0 --classes 12 --augment --refine --boundary-head 1.0 --no-test >> harth_bh10.log 2>&1
