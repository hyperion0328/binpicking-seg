#!/usr/bin/env bash
# 학습이 전부 끝난 뒤에 평가한다. GPU 를 동시에 쓰면 메모리 부족으로 죽는다(2회 겪음).
set -u
cd /home/user/sod-project
while pgrep -f '0[7]_trai[n]' >/dev/null || pgrep -f 'run_size_famil[y]' >/dev/null; do sleep 60; done
echo "########## 남은 런 평가 시작 ##########"
~/venvs/sod/bin/python scripts/08_eval.py \
  --only yolo11s-seg_640_e100,yolo26s-seg_640,yolo11m-seg_640 \
  --out outputs/d_eval_native.json 2>&1 | grep -avE "^\s*$"
echo "########## 평가 완료 ##########"
