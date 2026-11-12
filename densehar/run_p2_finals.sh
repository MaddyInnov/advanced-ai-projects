#!/bin/bash
cd ~/workspace/advanced-ai-projects/densehar
export OMP_WAIT_POLICY=PASSIVE
echo "=== confirm aug_c12 raw (TEST) ==="
python3 train_real.py confirm --model model_real_p2_aug_c12.pt --name final_augc12_raw
echo "=== postproc on aug_c12 (TEST) ==="
python3 train_real.py postproc --model model_real_p2_aug_c12.pt --name final_augc12_postproc
echo "=== postproc on aug_c12_refine (TEST) ==="
python3 train_real.py postproc --model model_real_p2_aug_c12_refine.pt --name final_augc12ref_postproc
echo "=== ensemble aug_c12 seeds 0,1,2 + postproc (TEST) ==="
python3 train_real.py ensemble --models model_real_p2_aug_c12.pt,model_real_p2_aug_c12_s1.pt,model_real_p2_aug_c12_s2.pt --name final_ens_augc12 --postproc-from final_augc12_postproc
echo "=== FINALS DONE ==="
