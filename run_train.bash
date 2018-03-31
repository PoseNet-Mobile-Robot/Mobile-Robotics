#/bin/bash
source activate tf27
python train.py > train_log.txt 2>&1 &
