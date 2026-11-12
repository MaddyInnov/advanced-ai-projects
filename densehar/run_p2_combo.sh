#!/bin/bash
cd ~/workspace/advanced-ai-projects/densehar
export OMP_WAIT_POLICY=PASSIVE
echo "=== p2_aug_refine ==="
python3 train_real.py exp --name p2_aug_refine --epochs 20 --augment --refine --no-test
echo "=== p2_aug_c12 ==="
python3 train_real.py exp --name p2_aug_c12 --epochs 20 --augment --classes 12 --no-test
echo "=== p2_aug_f11 ==="
python3 train_real.py exp --name p2_aug_f11 --epochs 20 --augment --features11 --no-test
echo "=== p2_aug30 ==="
python3 train_real.py exp --name p2_aug30 --epochs 30 --augment --no-test
echo "=== COMBO DONE ==="
