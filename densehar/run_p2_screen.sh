#!/bin/bash
cd ~/workspace/advanced-ai-projects/densehar
export OMP_WAIT_POLICY=PASSIVE
echo "=== postproc on p1_dense (val grid) ==="
python3 train_real.py postproc --model model_real_p1_dense.pt --name p2_postproc_p1 --no-test
echo "=== p2_plain (plain CE, no bw, no smooth) ==="
python3 train_real.py exp --name p2_plain --epochs 14 --no-bw --smooth 0 --no-test
echo "=== p2_smoothonly (T-MSE only) ==="
python3 train_real.py exp --name p2_smoothonly --epochs 14 --no-bw --smooth 0.15 --no-test
echo "=== p2_long (config A, 30 epochs) ==="
python3 train_real.py exp --name p2_long --epochs 30 --no-test
echo "=== p2_aug (physics augmentation, 20 epochs) ==="
python3 train_real.py exp --name p2_aug --epochs 20 --augment --no-test
echo "=== p2_refine (MS-TCN refinement stage, 20 epochs) ==="
python3 train_real.py exp --name p2_refine --epochs 20 --refine --no-test
echo "=== p2_c12 (12-class supervision, 20 epochs) ==="
python3 train_real.py exp --name p2_c12 --epochs 20 --classes 12 --no-test
echo "=== p2_cw (median-frequency class weights, 14 epochs) ==="
python3 train_real.py exp --name p2_cw --epochs 14 --class-weight --no-test
echo "=== p2_f11 (gravity+magnitude channels, 14 epochs) ==="
python3 train_real.py exp --name p2_f11 --epochs 14 --features11 --no-test
echo "=== SCREEN DONE ==="
