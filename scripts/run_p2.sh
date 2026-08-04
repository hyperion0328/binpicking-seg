#!/usr/bin/env bash
# P2 헤드 검증 - 2x2(헤드 x 마스크 해상도) x 2 크기.
# 기본 헤드 + mask_ratio 4 는 이미 있다(yolo11s-seg_160 / _320).
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
for sz in 160 320; do
  echo "########## 기본헤드 mr2 @ ${sz} ##########"
  $PY scripts/07_train.py --model yolo11s-seg.pt --imgsz ${sz} --mask-ratio 2 \
      --epochs 100 --batch 16 --name yolo11s-seg_${sz}_mr2 2>&1 | grep -avE "^\s*$"
  echo "########## P2 mr4 @ ${sz} ##########"
  $PY scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml --model yolo11s-seg.pt \
      --imgsz ${sz} --mask-ratio 4 --epochs 100 --batch 16 --name yolo11s-p2_${sz}_mr4 2>&1 | grep -avE "^\s*$"
  echo "########## P2 mr2 @ ${sz} ##########"
  $PY scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml --model yolo11s-seg.pt \
      --imgsz ${sz} --mask-ratio 2 --epochs 100 --batch 16 --name yolo11s-p2_${sz}_mr2 2>&1 | grep -avE "^\s*$"
done
echo "########## P2 검증 6회 완료 ##########"
