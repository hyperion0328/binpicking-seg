#!/usr/bin/env bash
# 결합 실험 3개 평가. 학습이 전부 끝난 뒤에만 돈다(GPU 겹쳐 쓰면 메모리 부족으로 죽는다).
set -u
cd /home/user/sod-project
while pgrep -f '0[7]_trai[n]' >/dev/null || pgrep -f 'run_comb[o]' >/dev/null; do sleep 60; done
echo "########## 결합 실험 평가 시작 ##########"
~/venvs/sod/bin/python scripts/08_eval.py \
  --only yolo11s-p2_320_deg,yolo11s-p2_640_deg,yolo11s-p2_320_deg_b16 \
  --out outputs/d_eval_native.json 2>&1 | grep -avE "^\s*$"
echo "########## 평가 완료 ##########"
