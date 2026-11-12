#!/bin/bash
cd ~/workspace/advanced-ai-projects/densehar
export OMP_WAIT_POLICY=PASSIVE
echo "=== p2_aug_f11 (fixed) ==="
python3 train_real.py exp --name p2_aug_f11 --epochs 20 --augment --features11 --no-test
echo "=== p2_aug_c12_refine ==="
python3 train_real.py exp --name p2_aug_c12_refine --epochs 20 --augment --classes 12 --refine --no-test
echo "=== p2_aug_c12_s1 ==="
python3 train_real.py exp --name p2_aug_c12_s1 --epochs 20 --augment --classes 12 --seed 1 --no-test
echo "=== p2_aug_c12_s2 ==="
python3 train_real.py exp --name p2_aug_c12_s2 --epochs 20 --augment --classes 12 --seed 2 --no-test
echo "=== COMBO2 DONE ==="
