#!/usr/bin/env bash
# 검증기 보정 전에 죽은 160 P2 두 런을 다시 돌린다. 앞 스크립트가 끝난 뒤 시작.
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
while pgrep -f run_p2.sh >/dev/null; do sleep 30; done
for mr in 4 2; do
  echo "########## (재실행) P2 mr${mr} @ 160 ##########"
  $PY scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml --model yolo11s-seg.pt \
      --imgsz 160 --mask-ratio ${mr} --epochs 100 --batch 16 --name yolo11s-p2_160_mr${mr} 2>&1 | grep -avE "^\s*$"
done
echo "########## P2 재실행 완료 ##########"
