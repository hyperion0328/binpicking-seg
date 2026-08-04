#!/usr/bin/env bash
# 주 곡선 6회 - n/s x 640/320/160. 고정 조건은 07_train.py 안에 있다.
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
for m in yolo11n-seg yolo11s-seg; do
  for sz in 640 320 160; do
    echo "########## ${m} @ ${sz} ##########"
    $PY scripts/07_train.py --model ${m}.pt --imgsz ${sz} --epochs 100 --batch 16 \
        --name ${m}_${sz} 2>&1 | grep -avE "^\s*$"
  done
done
echo "########## 곡선 6회 완료 ##########"
